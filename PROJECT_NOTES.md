# 📋 Chronic Kidney Disease (CKD) Classification — Project Notes

> **Purpose of this document:** A thorough technical walkthrough of every component in this project so that any developer — new or returning — can understand the structure, data flow, design decisions, bugs that were found and fixed, and how to run, maintain, or extend the project.

---

## Table of Contents

1. [Project at a Glance](#1-project-at-a-glance)
2. [Tech Stack & Dependencies](#2-tech-stack--dependencies)
3. [Folder Structure — Explained](#3-folder-structure--explained)
4. [The Dataset](#4-the-dataset)
5. [End-to-End Data & ML Flow](#5-end-to-end-data--ml-flow)
6. [Module-by-Module Breakdown](#6-module-by-module-breakdown)
7. [Model Artifacts](#7-model-artifacts)
8. [Test Suite](#8-test-suite)
9. [How to Run the Project](#9-how-to-run-the-project)
10. [Key Design Decisions](#10-key-design-decisions)
11. [Model Performance Results](#11-model-performance-results)
12. [Extension Ideas & Known Limitations](#12-extension-ideas--known-limitations)
13. [Bugs Found & Fixed During Submission Review](#13-bugs-found--fixed-during-submission-review)

---

## 1. Project at a Glance

This is a **supervised binary classification** machine learning project. It takes patient clinical lab data and predicts whether the patient has **Chronic Kidney Disease (CKD)** or not (`ckd` vs `notckd`).

| Property | Detail |
|---|---|
| **Problem type** | Binary classification |
| **Dataset source** | UCI Machine Learning Repository – CKD dataset |
| **Target variable** | `class` → values: `ckd` or `notckd` |
| **Dataset size** | ~200 rows × 29 columns |
| **Best model** | Random Forest Classifier |
| **Best model accuracy** | 100% on test set |
| **Best model ROC-AUC** | 1.0 (after bug fix) |
| **Interface** | Streamlit web app + CLI prediction |
| **Language** | Python 3.10+ |

---

## 2. Tech Stack & Dependencies

All dependencies are in `requirements.txt`.

| Library | Version | Role |
|---|---|---|
| `pandas` | >= 2.2.0 | Data loading, manipulation, and feature engineering |
| `numpy` | >= 1.26.0 | Numerical operations, NaN handling, array math |
| `scikit-learn` | >= 1.4.0 | ML pipelines, preprocessing transformers, models, metrics |
| `matplotlib` | >= 3.8.0 | Plotting confusion matrix, ROC curve, feature importance charts |
| `seaborn` | >= 0.13.0 | Imported (available for future visualizations) |
| `joblib` | >= 1.3.0 | Serializing (saving/loading) sklearn models and pipelines |
| `streamlit` | >= 1.32.0 | Interactive web UI for predictions |
| `pytest` | >= 8.0.0 | Automated unit/integration testing |

> **Virtual environment:** A `.venv/` folder is present. Always activate it before running anything.
> ```bash
> source .venv/bin/activate
> ```

---

## 3. Folder Structure — Explained

```
Kidney/
│
├── Dataset/                    <- Original raw CSV from UCI (do not modify)
│   └── ckd-dataset-v2.csv
│
├── data/
│   ├── raw/                    <- Where the pipeline reads raw data from
│   │   └── ckd_dataset.csv
│   └── processed/              <- Cleaned CSV saved after preprocessing (gitignored)
│       └── cleaned_dataset.csv
│
├── models/                     <- Trained model artifacts (committed to git)
│   ├── best_model.pkl          <- Serialized best classifier (Random Forest)
│   ├── preprocessing_pipeline.pkl  <- Serialized ColumnTransformer
│   └── model_metadata.json     <- Feature list, metrics, training timestamp
│
├── reports/
│   ├── figures/                <- Auto-generated charts
│   │   ├── feature_importance.png
│   │   ├── random_forest_confusion_matrix.png
│   │   └── random_forest_roc_curve.png
│   └── evaluation/             <- Auto-generated metrics JSON
│       └── metrics.json
│
├── src/                        <- Core Python package (all ML logic lives here)
│   ├── __init__.py
│   ├── config.py               <- Central path/constant definitions
│   ├── utils.py                <- Shared helper functions
│   ├── data_loader.py          <- Smart CSV loader with header detection
│   ├── preprocessing.py        <- Data cleaning and standardization
│   ├── feature_engineering.py  <- Domain-specific feature creation
│   ├── train.py                <- Full training pipeline (entry point)
│   ├── evaluate.py             <- Metrics, plots, and evaluation outputs
│   └── predict.py              <- Inference logic + CLI
│
├── app/
│   └── app.py                  <- Streamlit web application
│
├── notebooks/                  <- Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── tests/                      <- pytest test suite
│   ├── test_data.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
│
├── conftest.py                 <- ✅ Adds project root to sys.path for pytest (bug fix)
├── requirements.txt
├── README.md
├── DATASET.md
└── PROJECT_NOTES.md            <- This file
```

---

## 4. The Dataset

**Source:** UCI CKD Dataset — https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease

### Features (29 columns total: 28 input + 1 target)

| Feature | Full Name | Type |
|---|---|---|
| `age` | Age of patient | Numeric |
| `bp (diastolic)` | Blood Pressure (Diastolic) | Numeric |
| `bp limit` | Blood Pressure limit flag | Numeric |
| `sg` | Specific Gravity (urine) | Numeric |
| `al` | Albumin in urine | Numeric |
| `su` | Sugar in urine | Numeric |
| `rbc` | Red Blood Cells (urine) | Numeric |
| `pc` | Pus Cell (urine) | Numeric |
| `pcc` | Pus Cell Clumps | Numeric |
| `ba` | Bacteria | Numeric |
| `bgr` | Blood Glucose Random | Numeric |
| `bu` | Blood Urea | Numeric |
| `sod` | Sodium | Numeric |
| `sc` | Serum Creatinine | Numeric |
| `pot` | Potassium | Numeric |
| `hemo` | Hemoglobin | Numeric |
| `pcv` | Packed Cell Volume | Numeric |
| `rbcc` | Red Blood Cell Count | Numeric |
| `wbcc` | White Blood Cell Count | Numeric |
| `htn` | Hypertension flag | Numeric |
| `dm` | Diabetes Mellitus flag | Numeric |
| `cad` | Coronary Artery Disease flag | Numeric |
| `appet` | Appetite flag | Numeric |
| `pe` | Pedal Edema flag | Numeric |
| `ane` | Anemia flag | Numeric |
| `grf` | Glomerular Filtration Rate | Numeric |
| `stage` | CKD Stage (if known) | Numeric |
| `affected` | Affected flag | Numeric |
| **`class`** | **Target: `ckd` or `notckd`** | **Categorical** |

### Class Distribution (from latest training run)
- `ckd` → **128 samples**
- `notckd` → **72 samples**
- Dataset is **imbalanced** (~64% CKD). Both models are configured with `class_weight` to compensate.

### Raw Data Quirks
The CSV is NOT a clean, standard file. It contains:
- A **metadata/header row** at the top (not real column names)
- A second row describing variable types (e.g., "discrete", "continuous")
- Missing values encoded as `?`, blank strings, or `NaN`
- **Range-based values** like `< 0`, `1.019 - 1.021`, `112 - 154`

All of these quirks are handled by the data loader and preprocessing modules.

---

## 5. End-to-End Data & ML Flow

```
Raw CSV (Dataset/ckd-dataset-v2.csv)
         |
         v
  [data_loader.py] load_raw_dataset()
  -> Scores each row to detect the real header row (heuristic)
  -> Strips metadata rows and "discrete" type descriptor rows
  -> Normalizes column names (lowercase, stripped)
         |
         v
  [preprocessing.py] standardize_dataframe()
  -> Drops fully-empty columns and duplicate rows
  -> Cleans target labels (yes->ckd, not ckd->notckd, etc.)
  -> Parses ALL feature values via parse_numeric_value()
     (ranges -> median, "< X" -> X, "?" -> NaN)
  -> Keeps only rows with valid target labels (ckd / notckd)
         |
         v
  [feature_engineering.py] add_domain_features()
  -> Creates "age_group" (categorical age bins)
  -> Creates "hemoglobin_to_pcv_ratio" (hemo / pcv + epsilon)
         |
         v
  [train.py] build_preprocessor()
  -> Numeric columns  : SimpleImputer(strategy="median")
  -> Categorical cols : SimpleImputer(strategy="most_frequent")
                        + OneHotEncoder(handle_unknown="ignore")
  -> Combined via ColumnTransformer(remainder="drop")
         |
         v
  train_test_split (80% train / 20% test, stratified, random_state=42)
         |
         v
  Two candidate models trained and evaluated:
  +-------------------------------------------------+
  | 1. LogisticRegression                           |
  |    max_iter=2000, class_weight="balanced"       |
  |    solver="liblinear"                           |
  +-------------------------------------------------+
  | 2. RandomForestClassifier                       |
  |    n_estimators=300, random_state=42            |
  |    class_weight="balanced_subsample"            |
  |    min_samples_leaf=2                           |
  +-------------------------------------------------+
         |
         v
  [evaluate.py] compute_metrics() per model
  -> Accuracy, Precision, Recall, F1 (pos_label="ckd"), ROC-AUC
  -> P(ckd) correctly extracted using model.classes_ index lookup
  -> Best model selected by highest F1 score
         |
         v
  Artifacts saved to models/:
  -> best_model.pkl               (fitted classifier)
  -> preprocessing_pipeline.pkl   (fitted ColumnTransformer)
  -> model_metadata.json          (features, metrics, timestamp)
         |
         v
  Reports saved to reports/:
  -> figures/feature_importance.png
  -> figures/{model}_confusion_matrix.png
  -> figures/{model}_roc_curve.png
  -> evaluation/metrics.json
         |
         v
  INFERENCE (two interfaces):
  +-------------------------------+  +-----------------------------+
  | CLI: python -m src.predict   |  | Web: streamlit run app/app  |
  |      --input '{...json...}'  |  | Form-based Streamlit UI     |
  +-------------------------------+  +-----------------------------+
```

---

## 6. Module-by-Module Breakdown

---

### `src/config.py`

**Role:** Central configuration — all file paths and constants are defined here. No other module hardcodes paths.

```python
ROOT_DIR = Path(__file__).resolve().parents[1]   # -> Kidney/
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH       = DATA_DIR / "raw" / "ckd_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_dataset.csv"
MODELS_DIR          = ROOT_DIR / "models"
REPORTS_DIR         = ROOT_DIR / "reports"
FIGURES_DIR         = REPORTS_DIR / "figures"
EVALUATION_DIR      = REPORTS_DIR / "evaluation"
TARGET_COL                = "class"
DEFAULT_MODEL_PATH        = MODELS_DIR / "best_model.pkl"
DEFAULT_PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
DEFAULT_METADATA_PATH     = MODELS_DIR / "model_metadata.json"
```

> Uses `pathlib.Path` everywhere for cross-platform compatibility (Windows/macOS/Linux). All other modules import paths from here — changing a path only needs to happen in one place.

---

### `src/utils.py`

**Role:** Reusable helper functions shared across multiple modules.

| Function | What it does |
|---|---|
| `ensure_parent(path)` | Creates parent directories if they don't exist |
| `clean_text(value)` | Strips whitespace, converts "na" / "?" / "null" / "" to `None` |
| `normalize_category(value)` | Calls `clean_text`, then lowercases and replaces underscores with spaces |
| `parse_numeric_value(value)` | **Core function** — converts messy strings to floats |
| `save_json(data, path)` | Saves a dict to JSON file, auto-creates parent dirs |
| `load_json(path)` | Reads a JSON file and returns a dict |

#### How `parse_numeric_value` works

```
"?"           -> NaN
"< 0"         -> 0.0         (takes the number after the inequality)
"112 - 154"   -> 133.0       (takes the median of [112, 154])
"1.019-1.021" -> 1.02        (range -> median)
"1.019"       -> 1.019       (clean float -> returned as-is)
"nan"         -> NaN
""            -> NaN
```

This function handles all the messy real-world encodings in the UCI CKD dataset.

---

### `src/data_loader.py`

**Role:** Loads the raw CSV file, intelligently detecting which row is the actual header.

#### The header detection problem

The raw UCI CKD CSV is non-standard. It has metadata rows above the real feature names row. Instead of hardcoding `header=3`, the loader uses a **scoring heuristic**:

```python
def _row_header_score(row) -> int:
    # Known feature names list: ["bp (diastolic)", "sg", "al", "class", "age", ...]
    # For each known name found in the row: +2 points
    # Bonus for "class" being present:      +3 points
    # Bonus for "age" being present:        +2 points
    # Bonus for >= 20 non-empty values:     +2 points
    # The row with the highest score >= 8 is used as the header
```

After finding the header row:
- All rows above it are discarded
- Rows that are entirely "discrete" (type descriptor row) are removed
- Rows that are completely empty are removed
- Column names are normalized (`.strip().lower()`)

| Function | Purpose |
|---|---|
| `load_raw_dataset(path=None)` | Main loader — detects header, strips metadata, normalizes column names |
| `save_processed_dataset(df, path)` | Saves the cleaned DataFrame to CSV |

---

### `src/preprocessing.py`

**Role:** Cleans the loaded DataFrame — handles messy target labels and converts all feature values.

| Function | Purpose |
|---|---|
| `clean_target_labels(series)` | Normalizes messy class labels to exactly `"ckd"` or `"notckd"` |
| `standardize_dataframe(df)` | Full clean: drop all-null cols, drop duplicates, parse all features, filter to valid targets |
| `prepare_features(df, target_col)` | Splits DataFrame into `X` (features) and `y` (target Series) |

#### Target label normalization map

```
"ckd"     -> "ckd"
"notckd"  -> "notckd"
"not ckd" -> "notckd"
"no"      -> "notckd"
"no ckd"  -> "notckd"
"yes"     -> "ckd"
```

Any other value that isn't `NaN` is passed through; rows where the target is not exactly `"ckd"` or `"notckd"` are **dropped**.

---

### `src/feature_engineering.py`

**Role:** Creates new derived features that add predictive signal beyond what's in the raw columns.

| New Feature | Derived From | Logic |
|---|---|---|
| `age_group` | `age` | Bins: `child`(0-12), `teen`(12-20), `young_adult`(20-35), `adult`(35-50), `middle_aged`(50-65), `senior`(65+) |
| `hemoglobin_to_pcv_ratio` | `hemo`, `pcv` | `hemo / (pcv + 1e-6)` — epsilon prevents division-by-zero |

> **Why these features?** In nephrology, hemoglobin-to-PCV ratio is a clinical indicator related to anemia, which is itself a complication and sign of CKD. Age grouping lets the model capture non-linear age effects.

| Function | Purpose |
|---|---|
| `add_domain_features(df)` | Creates `age_group` and `hemoglobin_to_pcv_ratio`, returns updated DataFrame |
| `select_feature_columns(df, exclude)` | Utility — returns list of columns minus the excluded ones (for notebooks) |

---

### `src/train.py`

**Role:** Orchestrates the entire training pipeline. This is the **main entry point** for training.

```bash
python -m src.train
```

#### `build_preprocessor(X)` — The sklearn preprocessing pipeline

Automatically detects numeric vs. categorical columns in `X`:

```
Numeric columns  -> SimpleImputer(strategy="median")
Categorical cols -> SimpleImputer(strategy="most_frequent")
                   + OneHotEncoder(handle_unknown="ignore")
Combined via     -> ColumnTransformer(remainder="drop")
```

> **Why median for numerics?** Median is robust to outliers (e.g., extreme blood glucose). Mean would be skewed.
> **Why `handle_unknown="ignore"`?** Prevents crashes if an unseen category appears at inference time.

#### `train_and_evaluate()` — Execution order

1. `load_raw_dataset()` → raw DataFrame
2. `standardize_dataframe()` → cleaned DataFrame
3. `add_domain_features()` → adds `age_group`, `hemoglobin_to_pcv_ratio`
4. `save_processed_dataset()` → saves to `data/processed/`
5. `prepare_features()` → splits into X and y
6. `train_test_split(test_size=0.2, stratify=y, random_state=42)` → reproducible split
7. For each candidate model: wrap in `Pipeline`, fit, predict, compute metrics
8. **Best model selected by F1 score** (better than accuracy for imbalanced data)
9. Save preprocessor and model **separately** as `.pkl` files
10. Save `model_metadata.json` with feature list, all results, timestamps
11. Generate and save all evaluation charts

#### ✅ Bug Fix Applied Here (see Section 13, Bug #2)

P(ckd) is extracted using the `model.classes_` index lookup — not a hardcoded `[:, 1]`:

```python
classes = list(pipeline.named_steps["model"].classes_)
ckd_idx = classes.index("ckd") if "ckd" in classes else 1
y_prob = pipeline.predict_proba(X_test)[:, ckd_idx]
```

#### Candidate Models

| Model | Key Config |
|---|---|
| `LogisticRegression` | `max_iter=2000`, `class_weight="balanced"`, `solver="liblinear"` |
| `RandomForestClassifier` | `n_estimators=300`, `random_state=42`, `class_weight="balanced_subsample"`, `min_samples_leaf=2` |

---

### `src/evaluate.py`

**Role:** Computing metrics and generating all evaluation visualizations.

| Function | Output |
|---|---|
| `compute_metrics(y_true, y_pred, y_prob)` | Returns dict: accuracy, precision, recall, F1, ROC-AUC |
| `save_confusion_matrix(y_true, y_pred, path)` | Blue heatmap confusion matrix PNG |
| `save_feature_importance(model, preprocessor, path)` | Horizontal bar chart of top-15 features PNG |
| `save_metrics_report(metrics, path)` | Writes metrics dict to a JSON file |
| `save_roc_curve(y_true, y_prob, path)` | ROC curve plot PNG |
| `report_classification(y_true, y_pred, y_prob)` | Combines metrics + full sklearn `classification_report` dict |
| `create_evaluation_outputs(...)` | Orchestration — calls all of the above |

> **`matplotlib.use("Agg")`** is set at the top of the file before any other matplotlib import. This prevents matplotlib from trying to open a display window — essential for headless server environments or CI pipelines.

> **`pos_label="ckd"`** is set explicitly for precision, recall, and F1. This means metrics treat "ckd" as the positive (disease) class — the clinically correct choice.

---

### `src/predict.py`

**Role:** Runs inference on a single patient record. Usable as a Python function and as a CLI tool.

#### `predict_from_record(record, model_path, preprocessor_path)` — Core inference

1. Loads `model_metadata.json` to get the expected list of feature names
2. Loads preprocessor and model from `.pkl` files
3. Builds a DataFrame from the input `record` dict in exact training column order (missing features → `None` → NaN, handled by imputer)
4. Recomputes derived features (`age_group`, `hemoglobin_to_pcv_ratio`) if they are expected but not provided in the record
5. Transforms the row using the preprocessor
6. Runs `model.predict()` and `model.predict_proba()`
7. Returns a dict using `model.classes_` to correctly label probabilities

#### Return value format

```json
{
  "prediction": "ckd",
  "probability": {
    "ckd": 0.97,
    "notckd": 0.03
  },
  "target": "class"
}
```

> **Note:** `predict.py` correctly builds the probability map using `enumerate(model.classes_.tolist())` — this is safe and does NOT have the index bug that existed in `train.py`.

#### CLI usage

```bash
# Inline JSON:
python -m src.predict --input '{"age": 55, "bp (diastolic)": 80, ...}'

# JSON file:
python -m src.predict --input /path/to/patient.json
```

---

### `app/app.py`

**Role:** Streamlit web application — a user-friendly form to enter patient values and get predictions.

#### How it works

1. Loads `model_metadata.json` at startup to get the feature list
2. If model artifacts don't exist → shows warning + stops with `st.stop()`
3. Renders an `st.form` with `st.number_input` for each feature
4. Pre-filled clinically sensible defaults (age=55, hemo=12, sod=140, etc.)
5. Binary features (`htn`, `dm`, `cad`, `pe`, `ane`, `ba`, `pcc`) rendered with `min=0, max=10`
6. On form submit → calls `predict_from_record(record)` → displays result with `st.success` + `st.json`

```bash
streamlit run app/app.py
# Opens browser at http://localhost:8501
```

---

### `conftest.py` ✅ (Added as a bug fix)

**Role:** Root-level pytest configuration file. Adds the project root to `sys.path` automatically so that `from src.xxx import ...` works in all test files without any manual `PYTHONPATH` setup.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

> Without this file, running `pytest` from the project root would throw `ModuleNotFoundError: No module named 'src'` for every test. See Section 13, Bug #1.

---

## 7. Model Artifacts

All artifacts are auto-generated when you run `python -m src.train`. They are committed to git so that reviewers can use the Streamlit app and CLI without retraining.

| File | Contents |
|---|---|
| `best_model.pkl` | Fitted `RandomForestClassifier` (joblib-serialized) |
| `preprocessing_pipeline.pkl` | Fitted `ColumnTransformer` (imputers + OneHotEncoder, joblib-serialized) |
| `model_metadata.json` | Feature names list, training timestamp, all model results, best model name, class distribution |

> **Why saved separately?** The preprocessor and model are saved as independent files so either can be inspected, replaced, or versioned independently. `predict.py` loads and applies them in sequence: `.transform()` then `.predict()`.

---

## 8. Test Suite

Run tests with:

```bash
pytest -q    # concise output
pytest -v    # verbose output per test
```

> No `PYTHONPATH` setup needed — `conftest.py` handles it automatically (see Bug #1 fix).

| Test File | # Tests | What is tested |
|---|---|---|
| `test_data.py` | 3 | Raw CSV exists at expected path, has >0 rows and >20 columns, `class` column is present |
| `test_preprocessing.py` | 2 | `clean_target_labels` normalizes all known variants; `standardize_dataframe` correctly parses range values |
| `test_prediction.py` | 1 | `predict_from_record` returns a valid label (`ckd`/`notckd`) and correct probability keys |

> `test_prediction.py` uses `@pytest.mark.skipif` — it skips gracefully if model artifacts don't exist yet. Tests pass on a fresh clone before training.

**Final test result after all bug fixes: 6/6 passed ✅**

---

## 9. How to Run the Project

### First-time setup

```bash
# 1. Activate the virtual environment
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Confirm the raw dataset is in place
#    -> data/raw/ckd_dataset.csv must exist

# 4. Train the model (generates all artifacts and reports)
python -m src.train
```

### Run the Streamlit web app

```bash
streamlit run app/app.py
# Opens at http://localhost:8501
```

### Run CLI prediction

```bash
python -m src.predict --input '{"age": 55, "bp (diastolic)": 80, "bp limit": 0, "sg": 1.02, "al": 1, "su": 0, "rbc": 1, "pc": 0, "pcc": 0, "ba": 0, "bgr": 120, "bu": 50, "sod": 140, "sc": 1.2, "pot": 4.2, "hemo": 12, "pcv": 35, "rbcc": 4.7, "wbcc": 7500, "htn": 0, "dm": 0, "cad": 0, "appet": 0, "pe": 0, "ane": 0, "grf": 100, "stage": 0, "affected": 0}'
```

### Run tests

```bash
pytest -q
```

---

## 10. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Header detection via scoring heuristic** | Raw CSV has non-standard format. Heuristic avoids brittle hardcoded row numbers |
| **`class_weight` on both models** | Dataset is imbalanced (64% CKD / 36% notCKD). Prevents model from always predicting the majority class |
| **Best model selected by F1, not accuracy** | F1 is more meaningful for imbalanced data — accuracy can be misleadingly high |
| **Preprocessor + model saved separately** | Allows inspecting, swapping, or retraining either component independently |
| **Feature order enforcement in predict.py** | `_ensure_feature_order()` aligns input dict columns with training order, preventing silent misalignment bugs |
| **Derived features recomputed at inference** | `age_group` and `hemoglobin_to_pcv_ratio` are recreated from raw inputs — user doesn't need to compute them manually |
| **`model.classes_` index lookup for P(ckd)** | Instead of hardcoding `[:, 1]`, we find `"ckd"` in `model.classes_` to always get the correct probability column. This fixed Bug #2 |
| **`matplotlib.use("Agg")`** | Generates charts in headless environments (servers, CI/CD) without needing a display |
| **Median imputation for numerics** | Robust to outliers vs. mean; clinical values like blood glucose can be extreme |
| **`pathlib.Path` everywhere** | Cross-platform path handling (Windows / macOS / Linux) |
| **`conftest.py` at project root** | Automatically adds project root to `sys.path` so pytest works without any manual `PYTHONPATH` env var |

---

## 11. Model Performance Results

From `models/model_metadata.json` (latest training run, after all bug fixes):

| Metric | Logistic Regression | Random Forest (Best) |
|---|---|---|
| Accuracy | 97.5% | **100%** |
| Precision | 100% | **100%** |
| Recall | 96.15% | **100%** |
| F1 Score | 98.04% | **100%** |
| ROC-AUC | **99.45%** | **100%** |

> **Note on perfect scores:** 100% on 40 test samples is plausible given strong clinical feature signal and the small size of this clean dataset. Always validate on truly unseen patient data before any clinical deployment.

> **Note on ROC-AUC:** Before the bug fix, ROC-AUC was reported as `0.0` for Random Forest and `~0.005` for Logistic Regression due to an incorrect probability column being used. After fixing the `model.classes_` index lookup, the correct values are **1.0** and **0.9945** respectively.

---

## 12. Extension Ideas & Known Limitations

### Known Limitations

- **Small dataset:** Only ~200 rows. Models may not generalize well to diverse or unseen patient populations.
- **No cross-validation:** Single train/test split; K-fold cross-validation would give more reliable and less variance-prone performance estimates.
- **No input validation in Streamlit:** Invalid or extreme numerical inputs could produce silent errors or unexpected predictions.
- **Perfect test accuracy may be misleading:** The test set has only 40 samples — results should be interpreted cautiously for real-world use.

### Extension Ideas

- Add **K-fold cross-validation** in `train.py` using `sklearn.model_selection.cross_val_score`
- Add **more models** (XGBoost, SVM, KNN, GradientBoosting) to the `candidate_models` dict in `train.py`
- Add **SHAP explainability** for per-prediction feature attribution (shows which features drove the result)
- **Hyperparameter tuning** with `GridSearchCV` or `Optuna`
- **REST API** with FastAPI to replace or complement the CLI for integration with other systems
- **Docker containerization** for one-command reproducible deployment
- **CI/CD pipeline** (GitHub Actions) to auto-run `pytest` on every push/PR
- **Data versioning** with DVC for tracking dataset changes over time
- **Streamlit input validation** — add range checks and clinical bounds on input fields

---

## 13. Bugs Found & Fixed During Submission Review

During a final pre-submission audit, **3 bugs** were found and fixed. All are documented here so future maintainers understand what changed and why.

---

### Bug #1 — `pytest` Failed: `ModuleNotFoundError: No module named 'src'`

**Severity:** 🔴 Critical — no test could even be collected, let alone run.

**Symptom:**
```
ERROR tests/test_data.py
ModuleNotFoundError: No module named 'src'
ERROR tests/test_prediction.py
ModuleNotFoundError: No module named 'src'
ERROR tests/test_preprocessing.py
ModuleNotFoundError: No module named 'src'
```

Running `pytest` from the project root failed because Python's test runner did not know where `src/` was — the project root was not on `sys.path`.

**Root cause:**
There was no `conftest.py` at the project root. Without it, pytest does not automatically add the project directory to `sys.path`, so `from src.config import ...` fails.

**Fix:**
Created `conftest.py` at the project root:

```python
# conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

**Result:** `pytest -q` and `pytest -v` now work with zero setup — no manual `PYTHONPATH=.` needed.

**File changed:** `conftest.py` *(new file)*

---

### Bug #2 — ROC-AUC Was Completely Wrong (0.0 for Best Model)

**Severity:** 🔴 Critical — a key reported metric was factually incorrect.

**Symptom:**
`model_metadata.json` reported:
```json
{ "roc_auc": 0.0, "model_name": "random_forest" }
{ "roc_auc": 0.0054, "model_name": "logistic_regression" }
```

Both values are essentially random-chance level, which makes no sense for a model with 97–100% accuracy and F1.

**Root cause:**
In `src/train.py`, the probability column was always extracted with hardcoded index `[:, 1]`:

```python
# WRONG: assumes col 1 is P(ckd) — but model.classes_ = ['ckd', 'notckd']
y_prob = pipeline.predict_proba(X_test)[:, 1]
```

`model.classes_` is sorted alphabetically: `['ckd', 'notckd']`.
- Column index `0` = P(ckd) ← what we want
- Column index `1` = P(notckd) ← what we were using

So `roc_auc_score(y_true == "ckd", P(notckd))` computed AUC against the **wrong** class — essentially the complement of the correct AUC, hence ~0.0 instead of ~1.0.

**Fix:**
Replaced hardcoded `[:, 1]` with a dynamic class index lookup using `model.classes_`:

```python
# CORRECT: looks up the index of "ckd" in model.classes_
classes = list(pipeline.named_steps["model"].classes_)
ckd_idx = classes.index("ckd") if "ckd" in classes else 1
y_prob = pipeline.predict_proba(X_test)[:, ckd_idx]
```

The same fix was applied in both places in `train.py` (the training loop and the final evaluation call).

**Result after fix:**
```
random_forest:       ROC-AUC = 1.0    (was 0.0)
logistic_regression: ROC-AUC = 0.9945 (was 0.0054)
```

**File changed:** `src/train.py`

---

### Bug #3 — `.gitignore` Was Excluding Model Artifacts

**Severity:** 🟠 Major — anyone cloning the repo couldn't use the app without retraining.

**Symptom:**
The `.gitignore` file contained:
```
models/*.pkl
models/*.json
```

This meant `best_model.pkl`, `preprocessing_pipeline.pkl`, and `model_metadata.json` would **never be committed to git**. Any reviewer or user cloning the repo would get an empty `models/` directory, and the Streamlit app would immediately stop with "Train the project first...".

**Root cause:**
A well-intentioned but incorrect gitignore rule that treated model artifacts like build artifacts. For a small academic/portfolio project, committing the trained artifacts is the correct approach — it allows immediate use without retraining.

**Fix:**
Removed the model ignore rules from `.gitignore` and replaced with a comment explaining the decision. Also added `data/processed/` to gitignore (generated CSVs should not be committed):

```gitignore
# Before (wrong):
models/*.pkl
models/*.json

# After (correct):
data/processed/   # <- generated, large, reproducible from training

# Do NOT ignore models/ — pkl and json artifacts should be committed
# so reviewers can run the app without retraining
```

**File changed:** `.gitignore`

---

### Summary of All Fixes

| # | Bug | File(s) Changed | Impact |
|---|---|---|---|
| 1 | `pytest` broken — `ModuleNotFoundError: No module named 'src'` | `conftest.py` *(new)* | Tests now run with just `pytest -q` |
| 2 | ROC-AUC reported as 0.0 — wrong probability column used | `src/train.py` | AUC now correct: RF=1.0, LR=0.9945 |
| 3 | Model `.pkl` and `.json` files gitignored — app unusable after clone | `.gitignore` | Artifacts committed; app works immediately |

---

*Last updated: October 2026 | CKD Classification Project | Python 3.10+*
