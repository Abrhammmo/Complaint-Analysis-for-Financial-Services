import pandas as pd
import pytest
from src.sampling import stratified_sample

def test_stratified_sample_size():
    df = pd.DataFrame({
        "text": ["a"] * 100,
        "product": ["A"] * 50 + ["B"] * 50
    })

    sample = stratified_sample(df, "product", 20, random_state=42)
    assert len(sample) == 20

def test_stratified_sample_invalid_column():
    df = pd.DataFrame({"text": ["a", "b"]})
    with pytest.raises(ValueError):
        stratified_sample(df, "missing", 2, 42)
