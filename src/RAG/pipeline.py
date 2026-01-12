import pandas as pd

from .embedder import load_embedder, embed_query
from .vector_store import load_faiss_index, search
from .prompt import PROMPT_TEMPLATE
from .generator import load_generator, generate_answer

def run_rag(question, k=5):
    embedder = load_embedder()
    index, meta = load_faiss_index("../Data/processed/faiss_index")
    llm = load_generator()
    df = pd.read_csv("../Data/processed/sampled_complaints.csv")
    metadata_df = pd.DataFrame(meta)
    query_vec = embed_query(embedder, question)
    retrieved = search(index, query_vec, metadata_df, k)
    narratives = []
    for _, row in retrieved.iterrows():
        cid = row['complaint_id']
        narrative = df.iloc[cid]['Consumer complaint narrative']
        narratives.append(narrative)
    context = "\n\n".join(narratives[:3])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = generate_answer(llm, prompt)
    sources = []
    for _, row in retrieved.head(2).iterrows():
        cid = row['complaint_id']
        source = df.iloc[cid][['Product', 'Issue', 'Company']]
        sources.append(source.to_dict())
    return {
        "answer": response,
        "sources": sources
    }

