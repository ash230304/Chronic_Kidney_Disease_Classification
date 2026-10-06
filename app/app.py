import json
from pathlib import Path

import streamlit as st

from src.config import DEFAULT_METADATA_PATH, ROOT_DIR
from src.predict import predict_from_record
from src.utils import load_json

metadata = load_json(DEFAULT_METADATA_PATH) if Path(DEFAULT_METADATA_PATH).exists() else {"features": []}
feature_names = metadata.get("features", [])

if not feature_names:
    st.warning("Model metadata not found. Train the project first by running `python -m src.train`.")
    st.stop()

FEATURE_DEFAULTS = {
    "age": 55.0,
    "bp (diastolic)": 80.0,
    "bp limit": 0.0,
    "sg": 1.02,
    "al": 1.0,
    "su": 0.0,
    "rbc": 1.0,
    "pc": 0.0,
    "pcc": 0.0,
    "ba": 0.0,
    "bgr": 120.0,
    "bu": 50.0,
    "sod": 140.0,
    "sc": 1.2,
    "pot": 4.2,
    "hemo": 12.0,
    "pcv": 35.0,
    "rbcc": 4.7,
    "wbcc": 7500.0,
    "htn": 0.0,
    "dm": 0.0,
    "cad": 0.0,
    "appet": 0.0,
    "pe": 0.0,
    "ane": 0.0,
    "grf": 100.0,
    "stage": 0.0,
    "affected": 0.0,
}

st.set_page_config(page_title="CKD Predictor", layout="wide")
st.title("Chronic Kidney Disease (CKD) Classification")

with st.form("ckd_predictor"):
    record = {}
    for feat in feature_names:
        normalized_feature = feat.replace("_", " ")
        default = FEATURE_DEFAULTS.get(normalized_feature, FEATURE_DEFAULTS.get(feat, 0.0))
        if feat in {"bp limit", "rbc", "su", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "affected", "stage"}:
            record[feat] = st.number_input(feat, min_value=0.0, max_value=10.0, value=float(default))
        else:
            record[feat] = st.number_input(feat, value=float(default))
    submitted = st.form_submit_button("Predict")

if submitted:
    result = predict_from_record(record)
    st.success(f"Predicted class: {result['prediction']}")
    st.json(result)
