from src.loader import load
from src.chunker import chunk
from src.embedder import build_index, save_index

docs, _ = load("data")
chunks = chunk(docs)
store = build_index(chunks)
save_index(store)

print(f"Indexed {len(chunks)} chunks -> saved to index/")