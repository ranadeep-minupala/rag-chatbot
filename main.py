from dotenv import load_dotenv
load_dotenv()

from src.loader import load
from src.embedder import load_index
from src.bot import answer

print("Loading...")
_, business_info = load("data")
store = load_index("index")
print("Ready. Ask a question (type 'quit' to exit).\n")

while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() == "quit":
        break
    result = answer(question, store, business_info)
    print(f"Bot: {result['answer']}\n")