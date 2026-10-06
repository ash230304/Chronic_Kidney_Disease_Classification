# Chronic Kidney Disease Classification

This project builds a machine-learning pipeline to classify whether a patient is likely to have chronic kidney disease using the UCI CKD dataset. The project demonstrates a complete workflow from raw data ingestion and preprocessing to model evaluation, explainability, and prediction.

## Project goals
- Handle noisy clinical data with missing values and range-based values
- Train and compare several classification models
- Save the best model and preprocessing pipeline
- Provide a prediction interface via CLI and Streamlit app
- Create a portfolio-ready academic ML project

## Folder structure
```text
.
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── DATASET.md
├── README.md
├── requirements.txt
└── ...
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
```

## Run the prediction CLI
```bash
python -m src.predict --input '{"age": 55, "bp (diastolic)": 80, "bp limit": 0, "sg": 1.02, "al": 1, "su": 0, "rbc": 1, "pc": 0, "pcc": 0, "ba": 0, "bgr": 120, "bu": 50, "sod": 140, "sc": 1.2, "pot": 4.2, "hemo": 12, "pcv": 35, "rbcc": 4.7, "wbcc": 7500, "htn": 0, "dm": 0, "cad": 0, "appet": 0, "pe": 0, "ane": 0, "grf": 100, "stage": 0, "affected": 0}'
```

## Run Streamlit app
```bash
streamlit run app/app.py
```

## Validation
```bash
pytest -q
```

## Credits
The project builds on the public UCI CKD dataset and follows a standard academic ML workflow.
