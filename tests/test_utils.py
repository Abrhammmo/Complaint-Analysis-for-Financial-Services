import pandas as pd
import pytest
from src.utils import ensure_dir, validate_narratives

def test_ensure_dir(tmp_path):
    path = tmp_path / "new_dir"
    ensure_dir(path)
    assert path.exists()

def test_validate_narratives_raises():
    df = pd.DataFrame({"text": ["ok", None]})
    with pytest.raises(ValueError):
        validate_narratives(df, "text")
