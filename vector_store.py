import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity after normalization
        self.documents = []

    def add(self, embeddings, docs):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(docs)

    def search(self, query_embedding, k=12):
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append((score, self.documents[idx]))
        return results
