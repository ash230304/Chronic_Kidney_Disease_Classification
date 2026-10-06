import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in {"", "na", "n/a", "nan", "null", "none", "?"}:
        return None
    return text


def normalize_category(value):
    text = clean_text(value)
    if text is None:
        return None
    return text.lower().replace("_", " ").strip()


def parse_numeric_value(value):
    if pd.isna(value):
        return np.nan
    raw = clean_text(value)
    if raw is None:
        return np.nan
    raw = raw.replace(",", "")
    raw = raw.replace("≤", "<").replace("≥", ">")
    raw = raw.replace("µ", "u")

    if re.fullmatch(r"[-+]?\d*\.?\d+", raw):
        return float(raw)

    numbers = [float(token) for token in re.findall(r"[-+]?\d*\.?\d+", raw)]
    if not numbers:
        return np.nan

    if " - " in raw or "-" in raw and len(numbers) >= 2:
        return float(np.median(numbers))
    if "<" in raw or ">" in raw:
        return float(numbers[0])
    if len(numbers) == 1:
        return float(numbers[0])
    return float(np.median(numbers))


def save_json(data, path):
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
