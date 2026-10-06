from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH


def test_raw_dataset_exists():
    assert Path(RAW_DATA_PATH).exists()


def test_dataset_has_expected_shape():
    df = pd.read_csv(RAW_DATA_PATH, skiprows=[0, 1, 2])
    assert df.shape[0] > 0
    assert df.shape[1] > 20
    assert "class" in df.columns


def test_target_distribution_is_present():
    df = pd.read_csv(RAW_DATA_PATH, skiprows=[0, 1, 2])
    counts = df["class"].value_counts(dropna=False).to_dict()
    assert "ckd" in counts or "notckd" in counts
