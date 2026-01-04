from src.embeddings import embed_texts
import numpy as np

def test_embedding_shape():
    texts = ["hello world", "test sentence"]
    embeddings = embed_texts(texts, "sentence-transformers/all-MiniLM-L6-v2")

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384
