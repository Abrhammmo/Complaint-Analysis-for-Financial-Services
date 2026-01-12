# src/rag/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_embedder():
    return SentenceTransformer(_MODEL_NAME, local_files_only=True)

def embed_query(model, query: str) -> np.ndarray:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    vec = model.encode([query], normalize_embeddings=True)
    return vec.astype("float32")
