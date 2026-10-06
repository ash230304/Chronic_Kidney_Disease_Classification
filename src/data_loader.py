from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH


def _row_header_score(row: pd.Series) -> int:
    values = row.fillna("").astype(str).str.strip()
    lowered = values.str.lower()
    score = 0
    feature_names = ["bp (diastolic)", "bp limit", "sg", "al", "class", "rbc", "su", "pc", "pcc", "ba", "bgr", "bu", "sod", "sc", "pot", "hemo", "pcv", "rbcc", "wbcc", "htn", "dm", "cad", "appet", "pe", "ane", "grf", "stage", "affected", "age"]

    for feature in feature_names:
        if feature in lowered.values:
            score += 2

    if "class" in lowered.values:
        score += 3
    if "age" in lowered.values:
        score += 2
    if len(values[values != ""]) >= 20:
        score += 2
    if (lowered == "discrete").sum() == 0 and not values.empty:
        score += 1
    return score


def load_raw_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CKD dataset and recover the real header/data rows."""
    source = Path(path) if path is not None else RAW_DATA_PATH
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found at {source}")

    raw_df = pd.read_csv(source, header=None, dtype=str)
    if raw_df.empty:
        raise ValueError("Dataset is empty.")

    candidate_rows = []
    for index, row in raw_df.iterrows():
        score = _row_header_score(row)
        if score >= 8:
            candidate_rows.append((score, index, row))

    if not candidate_rows:
        raise ValueError("Could not detect the CKD dataset header row.")

    _, header_index, header_row = max(candidate_rows, key=lambda item: item[0])
    header = [str(value).strip() for value in header_row.tolist()]
    data = raw_df.iloc[header_index + 1 :].copy()
    data.columns = header

    data = data.loc[
        ~data.apply(
            lambda row: row.fillna("").astype(str).str.strip().str.lower().eq("discrete").all()
            or row.fillna("").astype(str).str.strip().eq("").all(),
            axis=1,
        )
    ].reset_index(drop=True)

    if "class" not in {str(col).strip().lower() for col in data.columns}:
        raise ValueError("Dataset header row does not contain the target column 'class'.")

    data.columns = [str(col).strip().lower() for col in data.columns]
    if data.empty:
        raise ValueError("Dataset is empty after removing malformed metadata rows.")
    return data


def save_processed_dataset(df: pd.DataFrame, output_path: str | Path) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return str(output)
