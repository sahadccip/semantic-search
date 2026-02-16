import requests
import re

# Ollama server URL
OLLAMA_API = "http://127.0.0.1:11434"

def rerank(query, documents):
    """
    Re-rank candidate documents using gemma:2b on Ollama.

    Args:
        query (str): Search query string.
        documents (list): List of dicts with keys: id, content, metadata.

    Returns:
        list: Documents with added 'score' key (0-1), sorted descending by score.
    """
    reranked = []

    for doc in documents:
        # Optimized prompt: numeric-only output
        prompt = f"""
You are a relevance scoring assistant.
Query: "{query}"
Document: "{doc['content']}"

On a scale from 0 (not relevant) to 10 (highly relevant), rate how relevant this document is to the query.
Respond with only a single number between 0 and 10. Nothing else.
"""

        # Call Ollama
        try:
            response = requests.post(
               f"{OLLAMA_API}/api/generate",
               json={
                "model": "gemma3:1b-it-qat",
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 10,
                "stream": False   # ⭐ IMPORTANT FIX
               },
               timeout=60
            )
            
            response.raise_for_status()
        except Exception as e:
            print(f"[ERROR] Ollama request failed: {e}")
            score = 0
        else:
            raw_json = response.json()
            # Handle streamed JSON response
            if isinstance(raw_json, list):
                full_text = "".join(chunk.get("response", "") for chunk in raw_json)
            else:
                full_text = raw_json.get("response", "")

            # Extract first number anywhere in response
            match = re.search(r"(\d+(\.\d+)?)", full_text.replace("\n", ""))
            if match:
                score = float(match.group(0))
            else:
                print(f"[WARN] Could not parse number from Ollama response: {full_text}")
                score = 0

        # Normalize score to 0-1
        normalized = min(max(score / 10, 0), 1)

        # Append score to document
        reranked.append({**doc, "score": normalized})

    # Sort descending by score
    reranked.sort(key=lambda x: x["score"], reverse=True)

    return reranked
