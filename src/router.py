import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM = """Classify the user's question into exactly one word:
- "TOOL" if it asks for a specific business fact (phone, email, PTO days, mileage rate, per diem, stipend amount, a number or contact).
- "RETRIEVE" if it asks about a policy, process, or how something works.
Reply with only the one word: TOOL or RETRIEVE."""

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
    return "TOOL" if "TOOL" in answer else "RETRIEVE"