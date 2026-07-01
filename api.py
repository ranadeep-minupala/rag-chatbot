from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from src.loader import load
from src.embedder import load_index
from src.bot import answer

app = FastAPI(title="Northwind RAG")

# Load once at startup
_, business_info = load("data")
store = load_index("index")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    return answer(query.question, store, business_info)

@app.get("/")
def health():
    return {"status": "ok"}