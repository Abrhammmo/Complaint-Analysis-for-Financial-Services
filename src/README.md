# Source Code

This directory contains the core source code for the Complaint Analysis for Financial Services project, implementing a Retrieval-Augmented Generation (RAG) system for analyzing customer complaints.

## Modules Overview

- `RAG/`: Contains components for the RAG pipeline.
  - `embedder.py`: Handles text embedding using Sentence Transformers.
  - `generator.py`: Loads and manages the language model for answer generation.
  - `pipeline.py`: Orchestrates the RAG process, from retrieval to generation.
  - `prompt.py`: Defines prompts for the language model.
  - `vector_store.py`: Manages FAISS vector search for efficient retrieval.
- `chunking.py`: Functions for splitting text into chunks.
- `config.py`: Configuration settings.
- `embeddings.py`: Embedding-related utilities.
- `faiss_store.py`: Utilities for building and persisting FAISS indexes.
- `sampling.py`: Data sampling functions.
- `utils.py`: General utility functions.
