
# 📊 Task 1: Exploratory Data Analysis & Data Preprocessing

## 📌 Objective

The goal of Task 1 is to explore, understand, and preprocess the CFPB consumer complaint dataset to ensure it is suitable for building a high-quality Retrieval-Augmented Generation (RAG) system. This task focuses on analyzing data structure, complaint distribution, narrative quality, and preparing clean textual inputs for downstream embedding and retrieval tasks.

---

## 📂 Dataset Description

This task uses the **full CFPB complaint dataset**, which contains real-world consumer complaints across multiple financial products. Each record includes structured metadata (e.g., product type, issue, company, state, date received) and an optional unstructured **Consumer complaint narrative**, which forms the core input for the RAG pipeline.

---

## 🔍 Key EDA Steps Performed

* Loaded and inspected the full CFPB complaint dataset to understand schema and data types.
* Analyzed complaint distribution across financial products to identify relevant categories.
* Examined the presence of complaint narratives and quantified records with missing or empty text.
* Calculated and visualized complaint narrative lengths (word count) using histograms, boxplots, and percentiles.
* Identified extremely short narratives that lack sufficient semantic context for embedding.
* Derived data-driven thresholds to guide filtering and preprocessing decisions.

---

## 🧹 Data Filtering & Preprocessing

To align with project requirements and improve embedding quality, the following preprocessing steps were applied:

* **Product Filtering**: Retained only complaints related to:

  * Credit Cards
  * Personal Loans
  * Savings Accounts
  * Money Transfers

* **Narrative Filtering**:

  * Removed complaints with missing or empty narratives.
  * Excluded narratives with fewer than 20 words based on percentile analysis.

* **Text Cleaning**:

  * Converted text to lowercase.
  * Removed boilerplate phrases commonly found in complaint openings.
  * Stripped special characters and normalized whitespace.

These steps ensure cleaner, more semantically meaningful text for embedding and retrieval.

---

## 📈 Key Findings

Here is a **clear, high-impact summary in a maximum of 5 bullet points**, suitable for a README, report, or slide deck:

* The CFPB dataset contains approximately **9.6 million complaints (2011–2025)**, with **credit-related issues dominating**, particularly *Credit reporting or other personal consumer reports*, which account for over **50%** of all submissions.
* Complaints are primarily submitted via **web-based channels**, with **Florida and California** recording the highest volumes, while only **~31%** of complaints include a consumer narrative, limiting qualitative analysis.
* Complaint narratives vary widely in length, with a **median of 100–150 words**, but over **20% are shorter than 20 words**, often consisting of boilerplate or incomplete text unsuitable for semantic retrieval.
* Frequently reported issues include **incorrect credit report information**, **account management problems**, and **transaction-related disputes**, indicating recurring challenges in financial services transparency.
* For RAG readiness, the data was filtered to **Credit Cards, Personal Loans, Savings Accounts, and Money Transfers**, cleaned through text normalization, and short narratives removed, resulting in a **high-quality dataset saved as `filtered_complaints.csv`** for downstream embedding and retrieval tasks.


---

## 📁 Outputs

* **Notebook**:
  `notebooks/task1_eda_preprocessing.ipynb`
  Contains all EDA, visualizations, and preprocessing steps.

* **Cleaned Dataset**:
  `data/filtered_complaints.csv`
  A filtered and cleaned dataset ready for embedding, vector indexing, and RAG development.

---

# 🧩 Task 2: Text Chunking, Embedding, and Vector Store Indexing

## 📌 Objective

The objective of Task 2 is to transform cleaned customer complaint narratives into a format suitable for **efficient semantic search**. This task prepares the foundation for a Retrieval-Augmented Generation (RAG) system by converting unstructured complaint text into vector embeddings and indexing them for fast similarity retrieval.

Due to hardware and time constraints, embeddings are generated on a **stratified sample** of the dataset while maintaining proportional representation across financial product categories.

---

## 📊 Stratified Sampling Strategy

To ensure balanced representation across complaint types, a **stratified sampling approach** was applied to the cleaned dataset produced in Task 1.

* Sample size: **~12,000 complaints**
* Stratification column: **Product category**
* Random seed fixed for reproducibility

This strategy guarantees that all target product categories (Credit Cards, Personal Loans, Savings Accounts, and Money Transfers) contribute proportionally to the embedding space, preventing dominance by high-volume categories.

The sampled dataset is persisted as:

```
data/processed/sampled_complaints.csv
```

---

## ✂️ Text Chunking Strategy

Customer complaint narratives vary significantly in length and structure. Embedding long narratives as a single vector can degrade retrieval performance. To address this, each narrative is split into overlapping text chunks.

* Chunking method: **Recursive character-based splitting**
* Chunk size: **500 characters**
* Chunk overlap: **50 characters**

This configuration balances semantic coherence with retrieval granularity while preserving contextual continuity across chunks.

---

## 🧠 Embedding Model Selection

The **`sentence-transformers/all-MiniLM-L6-v2`** model was selected for generating vector embeddings.

**Justification:**

* Produces high-quality sentence-level embeddings
* Compact (384-dimensional vectors)
* Fast inference suitable for CPU-only environments
* Widely adopted and well-validated for semantic search tasks

All embeddings are L2-normalized to enable cosine similarity via inner product search.

Generated embeddings are saved as:

```
data/processed/embeddings.npy
```

---

## 📦 Vector Store Indexing (FAISS)

To support fast and scalable similarity search, embeddings are indexed using **FAISS (Facebook AI Similarity Search)**.

* Index type: **IndexFlatIP (Inner Product)**
* Similarity metric: **Cosine similarity (via normalized vectors)**
* Offline, lightweight, and hardware-efficient

Metadata for each text chunk (complaint ID, product category, chunk index) is stored alongside the FAISS index to enable traceability and filtered retrieval.

Persisted artifacts:

```
data/processed/faiss_index/
├── index.faiss
└── index_meta.json
```

---

## 🏗 Modular & Reproducible Design

All core logic for Task 2 is implemented in the `src/` directory to promote modularity, testability, and reuse:

* Sampling
* Chunking
* Embedding
* FAISS indexing
* Validation utilities

A centralized `config.py` module defines all hyperparameters, paths, and random seeds, ensuring **deterministic and reproducible execution**.

The accompanying Jupyter notebook serves only as an orchestration layer, invoking well-defined functions from `src/` and documenting experimental decisions.

---

## 📁 Outputs Summary

After completing Task 2, the following artifacts are produced:

```
data/processed/
├── sampled_complaints.csv
├── embeddings.npy
└── faiss_index/
    ├── index.faiss
    └── index_meta.json
```

These outputs form the core semantic retrieval layer and will be used directly in **Task 3** to build and evaluate the RAG pipeline.

---

## ✅ Outcome

Task 2 establishes a robust, scalable, and reproducible semantic search foundation. Complaint narratives are now efficiently chunked, embedded, and indexed, enabling low-latency retrieval of relevant customer feedback for downstream question-answering and insight generation.

---

Below is a **clean, professional, and submission-ready README** covering **Task 3 and Task 4**, written to match **your actual implementation, constraints, and design choices**.
You can paste this **directly** into your project `README.md` or a `tasks/README.md`.

---

# 📌 Task 3 & Task 4 – RAG Core Logic, Evaluation & Interactive Interface

## Overview

Tasks 3 and 4 focus on transforming the preprocessed complaint data and vector embeddings into a **fully functional Retrieval-Augmented Generation (RAG) system** with an **interactive user interface**. The goal is to allow non-technical stakeholders at **CrediTrust Financial** to ask natural-language questions about customer complaints and receive **concise, evidence-backed answers** grounded in real customer narratives.

Due to computational constraints, the system operates on a **stratified, representative sample** created in Task 2, while preserving the same embedding model, chunking strategy, and retrieval logic as the full-scale dataset.

---

## ✅ Task 3: Building the RAG Core Logic & Evaluation

### 🎯 Objective

To design and implement a modular RAG pipeline that:

* Retrieves the most relevant complaint narratives using vector similarity search
* Generates grounded answers using an LLM
* Enables qualitative evaluation of retrieval and generation quality

---

### 🏗️ Architecture & Design

The RAG system is implemented in a **modular and reproducible manner**, with core logic placed under the `src/RAG/` directory and executed from a notebook.

```
src/RAG/
├── embedder.py        # Loads sentence embedding model
├── vector_store.py    # Loads FAISS index and performs similarity search
├── prompt.py          # Prompt template definition
├── generator.py       # Loads and runs the LLM
├── pipeline.py        # End-to-end RAG pipeline orchestration
```

The **notebook** is used only as an orchestration layer to:

* Load models and indexes
* Run sample queries
* Perform qualitative evaluation

---

### 🔍 Retriever Implementation

* **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`
* **Vector store**: FAISS (CPU)
* **Search method**: cosine similarity
* **Top-k retrieval**: `k = 5`
* **Metadata support**: each retrieved chunk includes product category, issue, company, and complaint narrative

Basic input validation ensures empty or invalid queries are safely handled.

---

### 🧠 Prompt Engineering

A structured prompt template is used to enforce **grounded generation**:

```
You are a financial analyst assistant for CrediTrust.
Use only the provided complaint excerpts to answer the question.
If the context does not contain enough information, say so.

Context:
{context}

Question:
{question}

Answer:
```

This design minimizes hallucination and ensures traceability between answers and source complaints.

---

### ✨ Generator Implementation

* **LLM backend**: Hugging Face Transformers (CPU)
* **Controlled generation**: max tokens and temperature tuned for concise answers
* **Graceful fallback**: informative message returned if generation fails

---

### 📊 Qualitative Evaluation

To evaluate system performance, **five representative business questions** were defined across product categories. For each query, the following were analyzed:

* Generated answer quality
* Relevance of retrieved complaint narratives
* Faithfulness to source content

An evaluation table was produced with the following columns:

* Question
* Generated Answer
* Retrieved Sources (1–2 examples)
* Quality Score (1–5)
* Comments / Analysis

This evaluation confirmed strong semantic retrieval and coherent, evidence-based answers, with minor limitations due to sample size.

---

## ✅ Task 4: Interactive Chat Interface

### 🎯 Objective

To build an intuitive, user-friendly web interface that enables **non-technical users** to interact with the RAG system in real time.

---

### 🖥️ Interface Implementation

* **Framework**: Gradio
* **Entry point**: `app.py`
* **Execution**: `python app.py`

---

### 🔑 Core Features

✔ Text input for user questions
✔ “Ask” button to submit queries
✔ Display area for AI-generated answers
✔ Source transparency via retrieved complaint excerpts
✔ “Clear” button to reset the interface

Each answer is accompanied by **verifiable source snippets**, increasing trust and interpretability.

---

### 🔄 System Flow

1. User enters a question in natural language
2. Question is embedded using the same model as the complaint embeddings
3. FAISS retrieves top-k relevant complaint chunks
4. Retrieved context is injected into the prompt
5. LLM generates a concise, grounded answer
6. Answer and sources are displayed in the UI

---

## ⚙️ Reproducibility & Robustness

* Fixed embedding model and retrieval parameters
* Modular source code with clear separation of concerns
* CPU-compatible implementation
* Graceful error handling for empty inputs and runtime failures
* Deterministic vector search behavior

---

## ⚠️ Limitations

* Operates on a **stratified sample** rather than the full 464K-complaint dataset due to memory constraints
* Generation quality depends on retrieved context richness
* No persistent chat history (single-turn Q&A)

---

## 🚀 Outcome

By the end of Tasks 3 and 4, the project delivers:

* A working RAG pipeline
* Transparent, evidence-based answers to complaint-related questions
* A production-ready interactive interface suitable for internal teams at CrediTrust

This foundation enables future scaling to the full dataset and deployment in real business workflows.



