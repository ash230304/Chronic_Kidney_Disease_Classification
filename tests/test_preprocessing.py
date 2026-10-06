import numpy as np
import pandas as pd

from src.preprocessing import clean_target_labels, standardize_dataframe


def test_clean_target_labels_normalizes_target():
    series = pd.Series(["ckd", " notckd ", "CKD", "not ckd", "?", "yes"])
    cleaned = clean_target_labels(series)
    assert set(cleaned.dropna().unique()) == {"ckd", "notckd"}


def test_standardize_dataframe_parses_numeric_ranges():
    df = pd.DataFrame({
        "class": ["ckd", "notckd"],
        "bp": ["< 0", "112 - 154"],
        "age": ["12 - 20", "< 12"],
    })
    cleaned = standardize_dataframe(df)
    assert not np.isnan(cleaned["bp"].iloc[0])
    assert not np.isnan(cleaned["age"].iloc[1])
