import os 
import sys

sys.path.append(os.path.abspath(os.path.join("..")))

import gradio as gr
import pandas as pd

from src.RAG.embedder import load_embedder
from src.RAG.generator import load_llm
from src.RAG.pipeline import run_RAG
from src.RAG.vector_store import load_faiss_index


# -----------------------------
# Load resources once
# -----------------------------
embedder = load_embedder()
llm = load_llm()

index, _ = load_faiss_index("Data/processed/faiss_index")

metadata_df = pd.read_csv(
    "Data/processed/sampled_complaints.csv"
)


# -----------------------------
# Chat handler
# -----------------------------
def chat(question):
    if not question or not question.strip():
        return "Please enter a valid question.", ""

    result = run_RAG(
        question=question,
        embedder=embedder,
        index=index,
        metadata_df=metadata_df,
        llm=llm,
        k=5
    )

    answer = result["answer"]

    sources_md = ""
    for i, row in result["sources"].iterrows():
        sources_md += f"""
**Source {i+1}**
- Product: {row['product_category']}
- Issue: {row['issue']}
- Company: {row['company']}

> {row['consumer_complaint_narrative'][:300]}...
---
"""

    return answer, sources_md


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown(
        "# 🏦 CrediTrust Complaint Analysis Assistant\n"
        "Ask questions about customer complaints across financial products."
    )

    question_box = gr.Textbox(
        label="Your Question",
        placeholder="Why are customers unhappy with credit cards?",
        lines=2
    )

    ask_button = gr.Button("Ask")
    clear_button = gr.Button("Clear")

    answer_box = gr.Markdown(label="Answer")
    sources_box = gr.Markdown(label="Sources")

    ask_button.click(
        fn=chat,
        inputs=question_box,
        outputs=[answer_box, sources_box]
    )

    clear_button.click(
        fn=lambda: ("", "", ""),
        inputs=None,
        outputs=[question_box, answer_box, sources_box]
    )
demo.launch(share=True)

