import requests
import numpy as np

OLLAMA_API = "http://localhost:11434"

def get_embedding(text: str):
    response = requests.post(
        f"{OLLAMA_API}/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return np.array(response.json()["embedding"], dtype="float32")
