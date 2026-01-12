# src/embedding.py

from sentence_transformers import SentenceTransformer
import numpy as np

def embed_texts(texts, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.array(embeddings, dtype="float32")

