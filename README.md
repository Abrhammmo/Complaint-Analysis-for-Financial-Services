
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

## Next steps
* **Task 3**: Building and evaluating the RAG core logic
* **Task 4**: Developing an interactive complaint-answering interface

