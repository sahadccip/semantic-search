from fastapi import FastAPI
from pydantic import BaseModel
import json
import time
import numpy as np

from embeddings import get_embedding
from vector_store import VectorStore
from reranker import rerank

app = FastAPI(title="Semantic Search API")

# ---------------- LOAD DOCUMENTS ----------------
with open("data/news.json", "r", encoding="utf-8") as f:
    DOCUMENTS = json.load(f)

# ---------------- BUILD VECTOR STORE (ON STARTUP) ----------------
print("Computing document embeddings...")

doc_embeddings = []
for doc in DOCUMENTS:
    emb = get_embedding(doc["content"])
    doc_embeddings.append(emb)

doc_embeddings = np.array(doc_embeddings).astype("float32")

dimension = doc_embeddings.shape[1]
vector_store = VectorStore(dimension)
vector_store.add(doc_embeddings, DOCUMENTS)

print("Vector store ready!")

# ---------------- REQUEST MODEL ----------------
class SearchRequest(BaseModel):
    query: str
    k: int = 12
    rerank: bool = True
    rerankK: int = 7

# ---------------- SEARCH ENDPOINT ----------------
@app.post("/search")
def search(req: SearchRequest):
    start_time = time.time()

    if not req.query.strip():
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": len(DOCUMENTS)
            }
        }

    # -------- STAGE 1: VECTOR SEARCH --------
    query_embedding = get_embedding(req.query)
    query_embedding = np.array([query_embedding]).astype("float32")

    results = vector_store.search(query_embedding, k=req.k)

    candidates = []
    for score, doc in results:
        # cosine similarity normalized from (-1,1) to (0,1)
        normalized_score = float((score + 1) / 2)

        candidates.append({
            "id": doc["id"],
            "content": doc["content"],
            "metadata": doc["metadata"],
            "score": normalized_score
        })

    # -------- STAGE 2: RE-RANK --------
    if req.rerank:
        reranked_results = rerank(req.query, candidates)
        final_results = reranked_results[:req.rerankK]
    else:
        final_results = candidates[:req.rerankK]

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": final_results,
        "reranked": req.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(DOCUMENTS)
        }
    }
