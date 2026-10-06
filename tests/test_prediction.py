import json
from pathlib import Path

import pytest

from src.config import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, DEFAULT_PREPROCESSOR_PATH
from src.predict import predict_from_record


@pytest.mark.skipif(not Path(DEFAULT_MODEL_PATH).exists() or not Path(DEFAULT_PREPROCESSOR_PATH).exists() or not Path(DEFAULT_METADATA_PATH).exists(), reason="Model artifacts do not exist yet")
def test_prediction_returns_valid_label():
    payload = {
        "age": 55,
        "bp (diastolic)": 80,
        "bp limit": 0,
        "sg": 1.02,
        "al": 1,
        "su": 0,
        "rbc": 1,
        "pc": 0,
        "pcc": 0,
        "ba": 0,
        "bgr": 120,
        "bu": 50,
        "sod": 140,
        "sc": 1.2,
        "pot": 4.2,
        "hemo": 12,
        "pcv": 35,
        "rbcc": 4.7,
        "wbcc": 7500,
        "htn": 0,
        "dm": 0,
        "cad": 0,
        "appet": 0,
        "pe": 0,
        "ane": 0,
        "grf": 100,
        "stage": 0,
        "affected": 0,
    }
    result = predict_from_record(payload)
    assert result["prediction"] in {"ckd", "notckd"}
    assert set(result["probability"].keys()) == {"ckd", "notckd"}
