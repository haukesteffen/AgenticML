"""v1_raw — trivial recipe: drop target from train, drop id from both."""
from __future__ import annotations

import pandas as pd


def build(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drop_train = [c for c in (target, id_column) if c in train_df.columns]
    drop_test = [c for c in (id_column,) if c in test_df.columns]
    return train_df.drop(columns=drop_train), test_df.drop(columns=drop_test)
