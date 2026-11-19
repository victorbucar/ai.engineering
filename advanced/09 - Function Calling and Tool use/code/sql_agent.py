"""Agent that combines local Python tools with a SQL Server MCP server.

Claude sees one flat tool list. A router sends each tool call either to a
local Python function or to the MCP server (started as a stdio child process).

Setup:
    pip install "anthropic[mcp]"
    Point SQL_SERVER below at your SQL Server MCP server (command, args, env
    var names come from that server's README). Use a read-only SQL login.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import anthropic
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.types import CallToolResult

from tools_implementations import get_weather, calculator


client = anthropic.AsyncAnthropic()
MODEL = "claude-haiku-5-5"
MAX_TURNS = 10

# --- Local tools ------------------------------------------------------------
LOCAL_TOOL_DEFS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a math expression with +, -, *, /, parentheses and decimals.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '(10 + 5) * 3'"},
                "precision": {"type": "integer", "description": "Decimal places, default 2"},
            },
            "required": ["expression"],
        },
    },
]
LOCAL_FUNCS = {"get_weather": get_weather, "calculator": calculator}

# --- SQL Server MCP server (sql-mcp, installed in its own venv) -------------
SQL_MCP_EXE = Path(__file__).parent / ".venv-sqlmcp" / "Scripts" / "sql-mcp.exe"
SQL_INSTANCE = r"DESKTOP-L6IGT74\SQLEXPRESS"
SQL_DATABASE = "VBTechBlog"

# Only these sql-mcp tools are offered to Claude (it exposes 30, incl. write tools).
ALLOWED_MCP_TOOLS = {
    "sql_list_tables",
    "sql_list_columns",
    "sql_list_foreign_keys",
    "sql_sample_rows",
    "sql_query",
}


def odbc_value(value: str) -> str:
    """Brace-quote an ODBC connection-string value so ; and } are safe."""
    return "{" + value.replace("}", "}}") + "}"


def sql_server_params() -> StdioServerParameters:
    user = os.environ.get("MSSQL_USER")
    password = os.environ.get("MSSQL_PASSWORD")
    if not user or not password:
        sys.exit("Set the MSSQL_USER and MSSQL_PASSWORD environment variables "
                 "(your read-only SQL login), then open a new terminal.")
    if not SQL_MCP_EXE.exists():
        sys.exit(f"sql-mcp not found at {SQL_MCP_EXE}")

    # A raw ODBC string (not host/port fields): sql-mcp would otherwise force
    # port 1433, which a named instance like SQLEXPRESS usually doesn't use.
    odbc = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SQL_INSTANCE};DATABASE={SQL_DATABASE};"
        f"UID={odbc_value(user)};PWD={odbc_value(password)};"
    )
    return StdioServerParameters(
        command=str(SQL_MCP_EXE),
        env={
            "SQL_URL": "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc),
            "SQL_ALLOW_WRITES": "False",  # sql-mcp's own read-only gate
            "SQL_MAX_ROWS": "100",
            "SQL_TIMEOUT_SECONDS": "30",
        },
    )


def run_local_tool(name: str, args: dict) -> tuple[str, bool]:
    try:
        out = LOCAL_FUNCS[name](**args)
    except TypeError as e:
        return json.dumps({"error": True, "message": f"Invalid arguments: {e}"}), True
    return json.dumps(out), bool(isinstance(out, dict) and out.get("error"))


async def run_mcp_tool(mcp: ClientSession, name: str, args: dict) -> tuple[str, bool]:
    result = await mcp.call_tool(name, args)
    # mcp 2.x can also return non-final results (e.g. InputRequiredResult)
    if not isinstance(result, CallToolResult):
        return f"MCP tool returned an unsupported result type: {type(result).__name__}", True
    text = "\n".join(c.text for c in result.content if c.type == "text")
    return text, bool(result.is_error)


async def run_agent(question: str) -> str:
    async with stdio_client(sql_server_params()) as (read, write):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()

            mcp_tools = [t for t in (await mcp.list_tools()).tools
                         if t.name in ALLOWED_MCP_TOOLS]
            mcp_names = {t.name for t in mcp_tools}
            missing = ALLOWED_MCP_TOOLS - mcp_names
            if missing:
                print(f"Warning: sql-mcp did not offer {sorted(missing)}")

            clash = mcp_names & LOCAL_FUNCS.keys()
            if clash:
                raise ValueError(f"Tool name collision between local and MCP tools: {clash}")

            tools = LOCAL_TOOL_DEFS + [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.input_schema,
                }
                for t in mcp_tools
            ]
            print(f"Tools: local={sorted(LOCAL_FUNCS)} mcp={sorted(mcp_names)}")

            messages = [{"role": "user", "content": question}]
            for _ in range(MAX_TURNS):
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    tools=tools,
                    messages=messages,
                )

                if response.stop_reason != "tool_use":
                    return next((b.text for b in response.content if b.type == "text"), "")

                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    if block.name in mcp_names:
                        content, is_error = await run_mcp_tool(mcp, block.name, block.input)
                        source = "mcp"
                    elif block.name in LOCAL_FUNCS:
                        content, is_error = run_local_tool(block.name, block.input)
                        source = "local"
                    else:
                        content, is_error = f"Unknown tool: {block.name}", True
                        source = "?"
                    print(f"  [{source}] {block.name}({json.dumps(block.input)}) error={is_error}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                        "is_error": is_error,
                    })

                messages += [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": results},
                ]

            return f"Stopped after {MAX_TURNS} tool turns without a final answer."


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "List the tables in the database, and tell me the weather in Tokyo."
    print(asyncio.run(run_agent(question)))
