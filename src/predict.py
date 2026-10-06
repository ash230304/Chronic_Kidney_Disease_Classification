from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import DEFAULT_MODEL_PATH, DEFAULT_PREPROCESSOR_PATH, DEFAULT_METADATA_PATH, TARGET_COL
from src.utils import load_json


def _ensure_feature_order(record: dict, feature_names: list[str]):
    row = {}
    for name in feature_names:
        if name in record:
            row[name] = record[name]
        else:
            row[name] = None
    return pd.DataFrame([row])


def predict_from_record(record: dict, model_path: str | Path = DEFAULT_MODEL_PATH, preprocessor_path: str | Path = DEFAULT_PREPROCESSOR_PATH) -> dict:
    metadata = load_json(DEFAULT_METADATA_PATH)
    feature_names = metadata.get("features", [])
    if not feature_names:
        raise ValueError("Model metadata does not contain feature names.")

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    frame = _ensure_feature_order(record, feature_names)

    if 'age_group' in frame.columns and 'age_group' not in record:
        frame['age_group'] = pd.cut(
            [float(record.get('age', 0))],
            bins=[0, 12, 20, 35, 50, 65, 100],
            labels=['child', 'teen', 'young_adult', 'adult', 'middle_aged', 'senior'],
            right=False
        )[0]
    if 'hemoglobin_to_pcv_ratio' in frame.columns and 'hemoglobin_to_pcv_ratio' not in record:
        hemo = float(record.get('hemo', 0.0) or 0.0)
        pcv = float(record.get('pcv', 0.0) or 0.0)
        frame['hemoglobin_to_pcv_ratio'] = hemo / (pcv + 1e-6)

    features = preprocessor.transform(frame)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    label = str(prediction)
    class_order = model.classes_.tolist()
    prob_map = {str(cls): float(probability[idx]) for idx, cls in enumerate(class_order)}
    return {
        "prediction": label,
        "probability": prob_map,
        "target": TARGET_COL,
    }


def predict_cli():
    parser = argparse.ArgumentParser(description="Predict chronic kidney disease risk from patient data." )
    parser.add_argument("--input", type=str, help="JSON string or path to a JSON file containing patient features.")
    args = parser.parse_args()

    payload = args.input
    if Path(payload).exists():
        with open(payload, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    else:
        record = json.loads(payload)
    print(json.dumps(predict_from_record(record), indent=2))


if __name__ == "__main__":
    predict_cli()
