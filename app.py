from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
import time

from reranker import rerank  # <- your updated reranker.py

app = FastAPI(title="Semantic Search API")

# Load your 65 news articles
with open("data/news.json", "r", encoding="utf-8") as f:
    DOCUMENTS = json.load(f)

# Request schema
class SearchRequest(BaseModel):
    query: str
    k: int = 12
    rerank: bool = True
    rerankK: int = 7

@app.post("/search")
def search(req: SearchRequest):
    start = time.time()
    
    # -------- Initial Retrieval --------
    # For now, simple retrieval: top-k by ID order
    # Replace this with real vector search if available
    candidates = DOCUMENTS[:req.k]

    # -------- Re-ranking --------
    if req.rerank:
        top_candidates = rerank(req.query, candidates)[:req.rerankK]
    else:
        top_candidates = candidates

    latency = int((time.time() - start) * 1000)  # ms

    return {
        "results": top_candidates,
        "reranked": req.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(DOCUMENTS)
        }
    }
