"""Multi-step agent: keeps calling tools until the model decides it is done.

Some requests need chained tool calls, where the input of one call comes from
the output of an earlier one, e.g.
    "Read the config file and tell me what model is configured,
     then search the web for that model's pricing."

Each iteration sends the whole conversation so far (every earlier tool call and
its result) back to the model, which then decides the next step: call more
tools, or answer. The loop stops when the model answers without calling a tool,
or after MAX_ITERATIONS model calls.
"""
import json
import sys
import time

import anthropic

from tool_registry import TOOL_REGISTRY, register_all_tools
from argument_validator import validate_tool_arguments


client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"
MAX_ITERATIONS = 10

SYSTEM = (
    "You are an assistant that can use tools. When a request needs several "
    "steps, call the tools one after another and use the results of earlier "
    "calls to decide the next call. When you have what you need, answer "
    "without calling more tools. If a tool returns an error or no results, "
    "say so plainly instead of guessing."
)


def anthropic_tools() -> list[dict]:
    """Convert the registry's function-style definitions to the API's tool format."""
    tools = []
    for entry in TOOL_REGISTRY.values():
        fn = entry["definition"]["function"]
        tools.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return tools


def execute_tool(name: str, args: dict) -> tuple[dict, float]:
    """Validate the arguments, run the tool, and return (result, elapsed_ms)."""
    errors = validate_tool_arguments(name, args)
    if errors:
        return {"error": True, "message": "; ".join(errors), "code": "INVALID_ARGUMENTS"}, 0.0

    start = time.perf_counter()
    try:
        result = TOOL_REGISTRY[name]["function"](**args)
    except Exception as e:  # a tool bug should become a tool error, not crash the loop
        result = {"error": True, "message": f"{type(e).__name__}: {e}"}
    return result, round((time.perf_counter() - start) * 1000, 2)


def run_agent(query: str, max_iterations: int = MAX_ITERATIONS, verbose: bool = True) -> dict:
    tools = anthropic_tools()
    messages = [{"role": "user", "content": query}]
    tool_log = []
    answer = None
    stop = "max_iterations"

    for iteration in range(1, max_iterations + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=tools,
            messages=messages,  # the full history: every earlier call and result
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # The model decided no more tools are needed (or it stopped for
            # another reason, e.g. max_tokens or refusal: reported in `stop`).
            answer = next((b.text for b in response.content if b.type == "text"), "")
            stop = response.stop_reason
            break

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result, elapsed_ms = execute_tool(block.name, block.input)
            is_error = bool(isinstance(result, dict) and result.get("error"))
            tool_log.append({
                "iteration": iteration,
                "tool": block.name,
                "arguments": block.input,
                "result": result,
                "is_error": is_error,
                "execution_time_ms": elapsed_ms,
            })
            if verbose:
                print(f"  [{iteration}] {block.name}({json.dumps(block.input)})"
                      f" -> {json.dumps(result)[:100]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
                "is_error": is_error,
            })

        # All results of this step go back in ONE user message
        messages.append({"role": "user", "content": results})

    return {
        "answer": answer,
        "stop": stop,
        "iterations": iteration,
        "tool_calls": tool_log,
        "messages": messages,
    }


if __name__ == "__main__":
    register_all_tools()
    query = " ".join(sys.argv[1:]) or (
        "Read the config file and tell me what model is configured, "
        "then search the web for that model's pricing."
    )
    print(f"User: {query}\n")
    result = run_agent(query)
    print(f"\nStop: {result['stop']} after {result['iterations']} iteration(s), "
          f"{len(result['tool_calls'])} tool call(s)")
    if result["answer"] is None:
        print(f"Stopped: reached the {MAX_ITERATIONS}-iteration limit without a final answer.")
    else:
        print(f"\nAssistant: {result['answer']}")
