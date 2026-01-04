from src.chunking import chunk_texts

def test_chunking_output():
    text = "This is a test sentence. " * 50
    chunks = chunk_texts(text, 100, 10)

    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
