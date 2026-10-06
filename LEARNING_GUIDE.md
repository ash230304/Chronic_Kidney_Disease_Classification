# 🧭 LEARNING_GUIDE.md
## How to Understand This Project as a New Person

> This guide walks you through the **exact order** to read, run, and explore this project so you can fully understand it in 1–2 hours — without getting lost.

---

## Table of Contents

1. [Before You Start — The Goal](#1-before-you-start--the-goal)
2. [Step 1 — Understand the Problem First](#step-1--understand-the-problem-first)
3. [Step 2 — Read the Config File](#step-2--read-the-config-file)
4. [Step 3 — Follow the Data Flow in Order](#step-3--follow-the-data-flow-in-order)
5. [Step 4 — Run the Project](#step-4--run-the-project)
6. [Step 5 — Look at the Generated Outputs](#step-5--look-at-the-generated-outputs)
7. [Step 6 — Read the Tests](#step-6--read-the-tests)
8. [Step 7 — Read the Bug Fix Section](#step-7--read-the-bug-fix-section)
9. [Recommended Reading Order for PROJECT_NOTES.md](#recommended-reading-order-for-project_notesmd)
10. [Quick Reference — Files at a Glance](#quick-reference--files-at-a-glance)
11. [TL;DR — The 1-Hour Path](#tldr--the-1-hour-path)

---

## 1. Before You Start — The Goal

This project answers one clinical question:

> **"Given a patient's lab test results, does the patient have Chronic Kidney Disease (CKD)?"**

The output is always one of two labels:
- `ckd` — the patient likely has Chronic Kidney Disease
- `notckd` — the patient likely does not

The project uses **real clinical data** from the UCI Machine Learning Repository (~200 patients, 28 lab features each) and trains a machine learning model to make this prediction automatically.

There are **two ways to use the trained model:**
1. A **command-line tool** — pass a JSON of patient values, get back a prediction
2. A **Streamlit web app** — fill a form in the browser, see the result instantly

---

## Step 1 — Understand the Problem First

**Time: ~5 minutes**

Read these files before you open any Python file:

| File | What to learn from it |
|---|---|
| `README.md` | What the project does, how to set it up, one-line commands |
| `DATASET.md` | What the data looks like, why it is tricky (non-standard CSV format) |
| `PROJECT_NOTES.md` -> Section 1 | One-table summary of the whole project |
| `PROJECT_NOTES.md` -> Section 4 | Full list of all 29 features with their clinical names |

### What you should know after this step:
- The **target column** is called `class`
- It has two values: `ckd` and `notckd`
- The dataset has ~200 rows, 28 input features
- The raw CSV is messy — it has metadata rows, `?` values, and range-based values like `112 - 154`

---

## Step 2 — Read the Config File

**Time: ~2 minutes**

Open `src/config.py`.

This is the **shortest file** in the project and the **most important one to read first**, because every other module imports paths and constants from it.

```python
ROOT_DIR              # -> the Kidney/ folder
RAW_DATA_PATH         # -> data/raw/ckd_dataset.csv
PROCESSED_DATA_PATH   # -> data/processed/cleaned_dataset.csv
MODELS_DIR            # -> models/
TARGET_COL            # -> "class"
DEFAULT_MODEL_PATH    # -> models/best_model.pkl
DEFAULT_PREPROCESSOR_PATH  # -> models/preprocessing_pipeline.pkl
DEFAULT_METADATA_PATH      # -> models/model_metadata.json
```

### What you should know after this step:
- Where every important file lives
- What `TARGET_COL = "class"` means (you will see this used everywhere)
- Why `pathlib.Path` is used instead of raw strings (cross-platform paths)

---

## Step 3 — Follow the Data Flow in Order

**Time: ~30-45 minutes**

Read the source files in **this exact order** — it matches the pipeline from raw data to trained model:

```
src/utils.py
    |
src/data_loader.py
    |
src/preprocessing.py
    |
src/feature_engineering.py
    |
src/train.py
    |
src/evaluate.py
    |
src/predict.py
    |
app/app.py
```

### What to ask yourself for each file:

> - What goes **IN** to this module?
> - What comes **OUT**?
> - What specific problem does it solve?

Use **Section 5** (the ASCII flow diagram) and **Section 6** (module breakdowns) in `PROJECT_NOTES.md` as a reference guide while you read each file.

---

### What each file is responsible for

#### `src/utils.py`
Helper functions used everywhere. The most important function here is `parse_numeric_value()`:

```
"?"           -> NaN       (missing value)
"< 0"         -> 0.0       (inequality value)
"112 - 154"   -> 133.0     (range value -> median)
"1.019"       -> 1.019     (clean float -> returned as-is)
```

This function is what converts the dataset's messy strings into actual numbers.

---

#### `src/data_loader.py`
The raw CSV is not a normal CSV — it has metadata rows before the actual data. This module uses a **scoring heuristic** to figure out which row is the real header:

```
For each row in the CSV:
  Count how many known feature names appear in it
  Row with the highest score is declared the header
  Everything above it is discarded
```

This is smarter than hardcoding `skiprows=3` — it works even if the file format changes slightly.

---

#### `src/preprocessing.py`
Cleans up the data after loading:
- Drops empty columns and duplicate rows
- Normalizes target labels (`"yes"` -> `"ckd"`, `"not ckd"` -> `"notckd"`, etc.)
- Runs `parse_numeric_value()` on every feature column
- Removes any row where the target is not `"ckd"` or `"notckd"`

---

#### `src/feature_engineering.py`
Creates two new features from existing ones:

| New Feature | How | Why |
|---|---|---|
| `age_group` | Bins `age` into child/teen/adult/etc. | Captures non-linear age effects |
| `hemoglobin_to_pcv_ratio` | `hemo / (pcv + tiny_epsilon)` | Clinical indicator related to anemia (a CKD complication) |

---

#### `src/train.py`
The **main entry point**. Orchestrates everything:
1. Loads -> cleans -> engineers features -> saves processed CSV
2. Splits data (80% train / 20% test, stratified)
3. Builds a preprocessing pipeline (imputer + encoder)
4. Trains two models: `LogisticRegression` and `RandomForestClassifier`
5. Picks the best one by **F1 score** (not accuracy — because the data is imbalanced)
6. Saves the model, preprocessor, metadata, and all charts

---

#### `src/evaluate.py`
Computes metrics and saves visualizations:
- Confusion matrix (heatmap PNG)
- ROC curve (PNG)
- Feature importance bar chart (PNG)
- Metrics JSON file

> Key detail: `pos_label="ckd"` is set on all metrics — this means "ckd" is treated as the positive (disease) class, which is the clinically correct choice.

---

#### `src/predict.py`
Takes a single patient's data (as a Python dict or JSON) and returns a prediction.
Works two ways:
- **As a function:** `predict_from_record(record)` — called by the Streamlit app
- **As a CLI tool:** `python -m src.predict --input '{"age": 55, ...}'`

> Important: It reads `model_metadata.json` to know which features and in what order they were trained on — this prevents silent column-mismatch bugs.

---

#### `app/app.py`
The Streamlit web UI. It:
1. Loads the feature list from `model_metadata.json`
2. Renders a number input for each feature (with sensible defaults pre-filled)
3. On submit, calls `predict_from_record()` and shows the result

---

## Step 4 — Run the Project

**Time: ~10 minutes**

Don't just read — actually run it. Seeing real output makes everything click.

```bash
# Activate the environment
source .venv/bin/activate

# Train the model (watch it produce models/ and reports/)
python -m src.train

# Run all tests (should see: 6 passed)
pytest -v

# Open the Streamlit app and make a prediction
streamlit run app/app.py
```

### What you should observe:
- `models/` folder fills with `.pkl` and `.json` files
- `reports/figures/` fills with chart images
- All 6 tests pass with green output
- The Streamlit UI opens at `http://localhost:8501` with a form

---

## Step 5 — Look at the Generated Outputs

**Time: ~10 minutes**

After training, open these files and trace them back to the code that created them:

| Generated File | Created by | Code location |
|---|---|---|
| `models/best_model.pkl` | `joblib.dump(model, ...)` | `train.py` |
| `models/preprocessing_pipeline.pkl` | `joblib.dump(preprocessor, ...)` | `train.py` |
| `models/model_metadata.json` | `save_json(metadata, ...)` | `train.py` |
| `reports/figures/feature_importance.png` | `save_feature_importance()` | `evaluate.py` |
| `reports/figures/random_forest_confusion_matrix.png` | `save_confusion_matrix()` | `evaluate.py` |
| `reports/figures/random_forest_roc_curve.png` | `save_roc_curve()` | `evaluate.py` |
| `reports/evaluation/metrics.json` | `save_metrics_report()` | `evaluate.py` |
| `data/processed/cleaned_dataset.csv` | `save_processed_dataset()` | `data_loader.py` |

Open `models/model_metadata.json` and look at the `"results"` and `"features"` keys — this is the source of truth the prediction module uses at inference time.

---

## Step 6 — Read the Tests

**Time: ~5 minutes**

Open `tests/` and read all three test files. Tests are often the **best short documentation** of what each module is supposed to do — they are faster to read than the source and state the contract clearly.

| Test File | What it tells you |
|---|---|
| `tests/test_data.py` | Raw CSV must exist at `RAW_DATA_PATH`, have >0 rows, >20 columns, and a `class` column |
| `tests/test_preprocessing.py` | `clean_target_labels` must normalize all variants; `standardize_dataframe` must parse range strings to numbers |
| `tests/test_prediction.py` | `predict_from_record` must return a label in `{"ckd", "notckd"}` and a `probability` dict with those exact keys |

Also look at `conftest.py` (in the project root) — it is a single small file that makes pytest work without any manual environment setup. Without it, you would get `ModuleNotFoundError: No module named 'src'` on every test.

---

## Step 7 — Read the Bug Fix Section

**Time: ~5 minutes**

Read **Section 13** of `PROJECT_NOTES.md` — "Bugs Found & Fixed During Submission Review".

This is where some of the **most valuable lessons** are, because each bug teaches a specific concept:

| Bug | Concept it teaches |
|---|---|
| Missing `conftest.py` -> all tests crash | How pytest resolves Python imports; what `sys.path` is and why it matters |
| ROC-AUC = 0.0 -> was using wrong column | How `model.classes_` ordering works; how `predict_proba` maps columns to labels; why you must never hardcode `[:, 1]` |
| `.gitignore` excluding `.pkl` files -> app broken after clone | What to commit in an ML project; difference between build artifacts and model artifacts |

Understanding these bugs will save you from making the same mistakes in your own projects.

---

## Recommended Reading Order for `PROJECT_NOTES.md`

You do not need to read `PROJECT_NOTES.md` all at once. Use it as a reference:

| When | Read | Purpose |
|---|---|---|
| First (before code) | Sections 1, 4, 5 | Big picture, dataset, pipeline flow diagram |
| While reading code | Section 6 | Module-by-module explanation alongside each file |
| After running | Sections 7, 11 | Artifact descriptions + actual metric numbers |
| For deeper understanding | Sections 10, 13 | Design decisions + bugs and why they happened |
| For future work | Section 12 | Limitations and what to build next |

---

## Quick Reference — Files at a Glance

```
Read first (no code):
  README.md                    -> Project overview and setup commands
  DATASET.md                   -> Dataset format and quirks
  PROJECT_NOTES.md             -> Full technical documentation (13 sections)
  LEARNING_GUIDE.md            -> This file — your step-by-step learning path

Config (read before any source file):
  src/config.py                -> All paths and constants

Pipeline (read in this order):
  src/utils.py                 -> Shared helpers (parse_numeric_value is key)
  src/data_loader.py           -> Loads and recovers the real header from raw CSV
  src/preprocessing.py         -> Cleans data, normalizes target labels
  src/feature_engineering.py   -> Creates age_group, hemoglobin_to_pcv_ratio
  src/train.py                 -> Trains models, selects best, saves artifacts
  src/evaluate.py              -> Computes metrics, generates charts
  src/predict.py               -> Runs inference on new patient data

Interface:
  app/app.py                   -> Streamlit web app (calls predict_from_record)

Tests (read these to understand what each module must do):
  tests/test_data.py
  tests/test_preprocessing.py
  tests/test_prediction.py

Pytest setup:
  conftest.py                  -> Adds project root to sys.path for pytest
```

---

## TL;DR — The 1-Hour Path

| # | Action | Time |
|---|---|---|
| 1 | Read `README.md` + `DATASET.md` | 5 min |
| 2 | Open `src/config.py` | 2 min |
| 3 | Read each `src/` file in pipeline order (utils -> data_loader -> preprocessing -> feature_engineering -> train -> evaluate -> predict -> app) | 35 min |
| 4 | Run `python -m src.train` then `pytest -v` | 5 min |
| 5 | Open `reports/` and `models/` output files | 5 min |
| 6 | Read `tests/` folder | 5 min |
| 7 | Read Section 13 of `PROJECT_NOTES.md` | 5 min |
| | **Total** | **~62 min** |

> After following this path you will understand every line of code in this project, why each design decision was made, what bugs were found and fixed, and how to extend or maintain the project going forward.

---

*Learning Guide — CKD Classification Project | Python 3.10+*
