# src/sampling.py

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    sample_size: int,
    random_state: int
) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        random_state=random_state
    )

    for train_idx, _ in splitter.split(df, df[label_col]):
        return df.iloc[train_idx].reset_index(drop=True)

