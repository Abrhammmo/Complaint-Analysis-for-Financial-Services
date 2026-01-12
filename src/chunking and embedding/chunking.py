# src/chunking.py

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_texts(texts, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(texts)

