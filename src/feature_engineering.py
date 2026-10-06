from __future__ import annotations

from typing import Iterable

import pandas as pd


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_frame = df.copy()
    if "age" in feature_frame.columns:
        feature_frame["age_group"] = pd.cut(
            feature_frame["age"],
            bins=[0, 12, 20, 35, 50, 65, 100],
            labels=["child", "teen", "young_adult", "adult", "middle_aged", "senior"],
            right=False,
        )
    if "hemo" in feature_frame.columns and "pcv" in feature_frame.columns:
        feature_frame["hemoglobin_to_pcv_ratio"] = feature_frame["hemo"] / (feature_frame["pcv"] + 1e-6)
    return feature_frame


def select_feature_columns(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> list[str]:
    exclude_set = set(exclude or [])
    return [col for col in df.columns if col not in exclude_set]
