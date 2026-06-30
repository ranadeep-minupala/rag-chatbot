import os, json
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.1-8b-instant"


def get_business_info(field, business_info):
    return business_info.get(field, "Not found")


def answer_with_tool(question, business_info):
    fields = list(business_info.keys())

    tools = [{
        "type": "function",
        "function": {
            "name": "get_business_info",
            "description": "Look up an exact business fact by field name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": fields}
                },
                "required": ["field"],
            },
        },
    }]

    messages = [
        {"role": "system", "content": "You are a company assistant. Use the get_business_info tool to look up facts, and answer using ONLY the tool result. Do not use outside knowledge."},
        {"role": "user", "content": question},
    ]

    # Round 1: model picks the field
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        temperature=0,
    )
    msg = resp.choices[0].message

    if not msg.tool_calls:
        return msg.content

    # Your code runs the lookup
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)
    result = get_business_info(args["field"], business_info)

    # Round 2: model phrases the answer using the result
    messages.append({
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [{
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": str(result),
    })

    final = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return final.choices[0].message.content