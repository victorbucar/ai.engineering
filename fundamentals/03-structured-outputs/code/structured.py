import os
import anthropic
import json
from datetime import date

message1 = (
    f"Today is {date.today():%A, %Y-%m-%d}. "
    "Extract the key information from this email: "
    "John Smith (john@example.com) is interested in our Enterprise plan "
    "and wants to schedule a demo for next Tuesday at 2pm."
)

def claude_call(message1):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,
        messages=[
            {
             "role": "user",
             "content": (
                 message1
                ),
         }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "plan_interest": {"type": "string"},
                        "demo_requested": {"type": "boolean"},
                        "date_time": {"type": "string", "format": "date-time"},
                    },
                    "required": ["name", "email", "plan_interest", "demo_requested", "date_time"],
                    "additionalProperties": False,
                },
            }
        },
    )

    text = next(block.text for block in response.content if block.type == "text")
    message = json.dumps(json.loads(text), indent=2)
    print("message from structured.py"+message)
    return message

if __name__ == "__main__":
    contact = claude_call(message1)
    
    # print(contact)