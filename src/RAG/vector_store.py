# src/rag/vector_store.py
import faiss
import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_faiss_index(index_dir: str):
    index_path = Path(index_dir) / "index.faiss"
    meta_path = Path(index_dir) / "index_meta.json"

    if not index_path.exists():
        raise FileNotFoundError("FAISS index not found")

    index = faiss.read_index(str(index_path))

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    return index, meta


def load_metadata(parquet_path: str):
    if not Path(parquet_path).exists():
        raise FileNotFoundError("Metadata parquet not found")

    return pd.read_parquet(parquet_path)


def search(index, query_vec, metadata_df, k: int = 5):
    if k <= 0:
        raise ValueError("k must be greater than 0")

    distances, indices = index.search(query_vec, k)

    results = metadata_df.iloc[indices[0]].copy()
    results["score"] = distances[0]

    return results
