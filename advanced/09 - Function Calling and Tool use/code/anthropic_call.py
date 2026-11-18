import anthropic
import json
from tools_implementations import get_weather


client = anthropic.Anthropic()

MODEL = "claude-haiku-5-5"
TOOLS = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
    }
}]

messages = [{"role": "user", "content": "Weather in Tokyo?"}]

response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    tools=TOOLS,
    messages=messages,
)

if response.stop_reason == "tool_use":
    results = []
    for block in response.content:
        if block.type == "tool_use":
            result = get_weather(**block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
                "is_error": bool(result.get("error")),
            })

    messages += [
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": results},
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )

print(next((b.text for b in response.content if b.type == "text"), ""))
