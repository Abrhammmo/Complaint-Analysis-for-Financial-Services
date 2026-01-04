import numpy as np
import os
from src.faiss_store import build_faiss_index, persist_faiss

def test_faiss_index_build(tmp_path):
    embeddings = np.random.rand(5, 384).astype("float32")

    index = build_faiss_index(embeddings)
    assert index.ntotal == 5

def test_faiss_persistence(tmp_path):
    embeddings = np.random.rand(3, 384).astype("float32")
    index = build_faiss_index(embeddings)

    metadata = [{"id": i} for i in range(3)]
    persist_faiss(index, metadata, tmp_path)

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "index_meta.json").exists()
