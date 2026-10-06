from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ckd_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_DIR = REPORTS_DIR / "evaluation"
TARGET_COL = "class"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.pkl"
DEFAULT_PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
DEFAULT_METADATA_PATH = MODELS_DIR / "model_metadata.json"
