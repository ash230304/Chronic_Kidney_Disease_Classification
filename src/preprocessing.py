from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COL
from src.utils import normalize_category, parse_numeric_value


def clean_target_labels(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower().replace({"nan": np.nan, "?": np.nan})
    mapping = {
        "ckd": "ckd",
        "notckd": "notckd",
        "not ckd": "notckd",
        "no": "notckd",
        "no ckd": "notckd",
        "yes": "ckd",
    }
    result = normalized.map(mapping).fillna(normalized)
    return result.replace({"": np.nan})


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates().reset_index(drop=True)

    for column in df.columns:
        if column == TARGET_COL:
            df[column] = clean_target_labels(df[column])
            continue
        df[column] = df[column].apply(parse_numeric_value)

    df = df[df[TARGET_COL].isin(["ckd", "notckd"])].reset_index(drop=True)
    df = df.dropna(axis=0, subset=[TARGET_COL]).reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> tuple[pd.DataFrame, pd.Series]:
    feature_frame = df.drop(columns=[target_col], errors="ignore")
    target = df[target_col]
    return feature_frame, target
