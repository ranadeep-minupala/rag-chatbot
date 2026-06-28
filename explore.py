from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "You can return items within 30 days for a full refund.",
    "Our store hours are 9am to 5pm, Monday through Friday.",
    "Shipping is free on orders over $50.",
    "Contact support at help@example.com.",
    "Mileage is reimbursed at 65 cents per mile.",
]

# 1. Embed all sentences.  model.encode(...) takes a list, returns vectors.
embeddings = model.encode(sentences)

# 2. Embed the question.  (same method, give it the one string)
question = "what are timings?"
q_vec = model.encode(question)

# 3. Write cosine yourself — don't import it.
#    cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
#    hints: np.dot(a, b)   and   np.linalg.norm(a)

def cosine(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# 4. Loop: score the question against each sentence, collect (score, sentence),
#    sort highest-first, then print each line.
#    hint: zip(sentences, embeddings) pairs each sentence with its vector.
scores = []
for sentence, vec in zip(sentences, embeddings):
    s = cosine(q_vec, vec)
    scores.append((s, sentence))

scores.sort(reverse=True)   # highest first

for s, sentence in scores:
    print(f"{s:.3f}  {sentence}")