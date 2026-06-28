import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

class VectorStore:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def search(self, query, k=5):
        q = model.encode([query])                      # embed the query
        q = np.array(q, dtype="float32")
        faiss.normalize_L2(q)                          # normalize for cosine
        scores, ids = self.index.search(q, k)          # FAISS returns top-k
        return [self.chunks[i] for i in ids[0]]        # map ids back to chunks

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts)                      # embed all chunks
    vectors = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vectors)                        # normalize for cosine

    dim = vectors.shape[1]                             # 384
    index = faiss.IndexFlatIP(dim)                     # inner product = cosine (normalized)
    index.add(vectors)

    return VectorStore(index, chunks)