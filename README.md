
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

## 🚀 Next Steps

The cleaned dataset produced in Task 1 will be used in:

* **Task 2**: Text chunking, embedding generation, and vector store indexing
* **Task 3**: Building and evaluating the RAG core logic
* **Task 4**: Developing an interactive complaint-answering interface

