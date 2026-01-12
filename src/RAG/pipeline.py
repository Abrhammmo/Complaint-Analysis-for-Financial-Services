import pandas as pd

from .embedder import load_embedder, embed_query
from .vector_store import load_faiss_index, search
from .prompt import PROMPT_TEMPLATE
from .generator import load_llm, generate_answer

def run_RAG(question, embedder, index, metadata_df, llm, k=5):
    query_vec = embed_query(embedder, question)
    retrieved = search(index, query_vec, metadata_df, k)
    narratives = []
    for _, row in retrieved.iterrows():
        narrative = row['consumer_complaint_narrative']
        narratives.append(narrative)
    context = "\n\n".join(narratives[:3])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = generate_answer(llm, prompt)
    sources = []
    for _, row in retrieved.head(2).iterrows():
        source = row[['product_category', 'issue', 'company']]
        sources.append(source.to_dict())
    return {
        "answer": response,
        "sources": pd.DataFrame(sources)
    }

