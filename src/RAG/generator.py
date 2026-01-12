# src/rag/generator.py
from transformers import pipeline

def load_generator(model_name="google/flan-t5-base"):
    return pipeline(
        "text2text-generation",
        model=model_name,
        max_new_tokens=256
    )

def generate_answer(llm, prompt: str) -> str:
    output = llm(prompt)
    return output[0]["generated_text"]
