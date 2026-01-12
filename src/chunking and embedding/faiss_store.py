# src/faiss_store.py

import faiss
import json
import os
import numpy as np

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def persist_faiss(index, metadata, path):
    os.makedirs(path, exist_ok=True)

    faiss.write_index(index, f"{path}/index.faiss")

    with open(f"{path}/index_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

