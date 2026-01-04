# src/config.py

class Config:
    RANDOM_SEED = 42

    # Sampling
    SAMPLE_SIZE = 12000

    # Chunking
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Embeddings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Paths
    PROCESSED_DIR = "../data/processed/"
    FAISS_DIR = "../data/processed/faiss_index"