# src/utils.py

import os
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def validate_narratives(df, text_col):
    if df[text_col].isnull().any():
        raise ValueError("Null narratives detected")
