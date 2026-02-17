import requests
import numpy as np

OLLAMA_API = "http://127.0.0.1:11434"

def get_embedding(text: str):
    response = requests.post(
        f"{OLLAMA_API}/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )

    response.raise_for_status()
    embedding = response.json()["embedding"]

    return np.array(embedding, dtype="float32")
