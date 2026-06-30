import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM = """Classify the user's message into exactly one word.

- "TOOL" if it asks for a specific business fact, even as a single keyword.
  Examples: "mileage rate", "hr email", "phone number", "address", "pto days", "per diem".
- "RETRIEVE" if it asks about a policy, process, or how something works.
  Examples: "how do I request leave?", "what is the remote work policy?", "expense rules".
- "CLARIFY" only if the message is a greeting, conversational filler, off-topic,
  or too vague to map to any company fact or policy.
  Examples: "hi", "how are you", "thanks", "claude", "weather", "okay bye".

Prefer TOOL or RETRIEVE when the message plausibly refers to a company fact or policy.
Use CLARIFY only when it clearly does not.

Reply with only the one word: TOOL, RETRIEVE, or CLARIFY."""


def classify(question):
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip().upper()
    for route in ("TOOL", "RETRIEVE", "CLARIFY"):
        if route in answer:
            return route
    return "CLARIFY"