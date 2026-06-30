import os
from groq import Groq
from src.router import classify
from src.tool import answer_with_tool

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.1-8b-instant"

SYSTEM = """You are a company policy assistant. Answer the question using ONLY the context below. If the answer is not in the context, say "I don't have that information." Do not use outside knowledge."""


def answer_with_retrieval(question, store):
    chunks = store.search(question, k=5)
    context = "\n\n".join(c["text"] for c in chunks)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content, chunks


def answer(question, store, business_info):
    route = classify(question)

    if route == "TOOL":
        text = answer_with_tool(question, business_info)
        return {"answer": text, "route": route, "sources": []}

    text, chunks = answer_with_retrieval(question, store)
    sources = [c["source"] for c in chunks]
    return {"answer": text, "route": route, "sources": sources}