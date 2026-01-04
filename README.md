
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

* A significant portion of complaints lack narratives and are unsuitable for semantic search.
* Narrative lengths vary widely, from very short descriptions to detailed multi-paragraph complaints.
* Extremely short narratives fall below the 5th percentile and provide limited contextual value.
* The observed distribution of long narratives validates the need for chunking in subsequent tasks.

---

## 📁 Outputs

* **Notebook**:
  `notebooks/task1_eda_preprocessing.ipynb`
  Contains all EDA, visualizations, and preprocessing steps.

* **Cleaned Dataset**:
  `data/filtered_complaints.csv`
  A filtered and cleaned dataset ready for embedding, vector indexing, and RAG development.

---

The exploratory data analysis (EDA) of the CFPB consumer complaints dataset revealed several important insights into complaint patterns, data quality, and narrative usability. The dataset contains approximately 9.6 million complaints spanning from 2011 to 2025, with credit-related issues dominating the submissions. The product category “Credit reporting or other personal consumer reports” alone accounts for nearly 4.8 million complaints, representing over 50% of the total volume, with major credit bureaus such as Equifax frequently cited. Most complaints were submitted via web-based channels, and Florida and California emerged as the states with the highest complaint volumes. Despite the large dataset size, only about 31% of complaints include a consumer complaint narrative, highlighting a substantial gap in qualitative, context-rich data.

A deeper analysis of narrative quality revealed significant variability in text length and content. Among complaints with narratives, word counts ranged widely, with a median length of approximately 100–150 words. However, more than 20% of narratives contained fewer than 20 words, often consisting of boilerplate phrases or incomplete descriptions. Such short narratives provide limited semantic value and are unsuitable for embedding-based retrieval systems. Commonly reported issues included incorrect information on credit reports, account management problems, and transaction-related disputes, pointing to recurring systemic challenges in financial services transparency and consumer protection.

To prepare a high-quality dataset suitable for Retrieval-Augmented Generation (RAG) and advanced analytics, the data was filtered to four target product categories: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers. Complaints without narratives were removed, and additional text preprocessing steps were applied, including boilerplate removal, lowercasing, special character stripping, and whitespace normalization. Narratives shorter than 20 words were excluded based on percentile analysis. These steps resulted in a refined, narrative-rich dataset saved as filtered_complaints.csv, ready for downstream tasks such as embedding, semantic retrieval, topic modeling, and sentiment analysis.

---

## 🚀 Next Steps

The cleaned dataset produced in Task 1 will be used in:

* **Task 2**: Text chunking, embedding generation, and vector store indexing
* **Task 3**: Building and evaluating the RAG core logic
* **Task 4**: Developing an interactive complaint-answering interface

