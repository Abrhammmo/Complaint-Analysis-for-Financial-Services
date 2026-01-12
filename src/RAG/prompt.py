# src/rag/prompt.py

PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.

Use ONLY the information provided in the context below.
If the context does not contain enough information, say so explicitly.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def build_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
