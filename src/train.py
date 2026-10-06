from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import MODELS_DIR, PROCESSED_DATA_PATH, REPORTS_DIR, TARGET_COL
from src.data_loader import load_raw_dataset, save_processed_dataset
from src.evaluate import compute_metrics, create_evaluation_outputs, save_feature_importance
from src.feature_engineering import add_domain_features
from src.preprocessing import prepare_features, standardize_dataframe
from src.utils import save_json


def build_preprocessor(X: pd.DataFrame):
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            numeric_columns,
        ))
    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_columns,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_and_evaluate():
    raw_df = load_raw_dataset()
    cleaned_df = standardize_dataframe(raw_df)
    cleaned_df = add_domain_features(cleaned_df)
    save_processed_dataset(cleaned_df, PROCESSED_DATA_PATH)

    X, y = prepare_features(cleaned_df, target_col=TARGET_COL)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidate_models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
    }

    results = []
    best_model_name = None
    best_pipeline = None
    best_metrics = None

    for model_name, model in candidate_models.items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor(X_train)), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        # Use the index of "ckd" in model.classes_ so P(ckd) is always used
        # regardless of alphabetical ordering. Falls back to None if unavailable.
        if hasattr(pipeline, "predict_proba"):
            classes = list(pipeline.named_steps["model"].classes_)
            ckd_idx = classes.index("ckd") if "ckd" in classes else 1
            y_prob = pipeline.predict_proba(X_test)[:, ckd_idx]
        else:
            y_prob = None
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["model_name"] = model_name
        results.append(metrics)

        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_model_name = model_name
            best_pipeline = pipeline
            best_metrics = metrics

    if best_pipeline is None:
        raise RuntimeError("No model was trained successfully.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline.named_steps["preprocessor"], MODELS_DIR / "preprocessing_pipeline.pkl")
    joblib.dump(best_pipeline.named_steps["model"], MODELS_DIR / "best_model.pkl")

    original_feature_names = list(X_train.columns)
    metadata = {
        "target_column": TARGET_COL,
        "best_model": best_model_name,
        "training_time_utc": datetime.now(timezone.utc).isoformat(),
        "features": original_feature_names,
        "results": results,
        "best_metrics": best_metrics,
        "test_size": 0.2,
        "class_distribution": {k: int(v) for k, v in y.value_counts().to_dict().items()},
    }
    save_json(metadata, MODELS_DIR / "model_metadata.json")
    save_feature_importance(best_pipeline.named_steps["model"], best_pipeline.named_steps["preprocessor"], REPORTS_DIR / "figures" / "feature_importance.png")
    best_classes = list(best_pipeline.named_steps["model"].classes_)
    best_ckd_idx = best_classes.index("ckd") if "ckd" in best_classes else 1
    best_y_prob = best_pipeline.predict_proba(X_test)[:, best_ckd_idx] if hasattr(best_pipeline, "predict_proba") else None
    create_evaluation_outputs(y_test, best_pipeline.predict(X_test), best_y_prob, best_model_name, REPORTS_DIR / "evaluation")
    return cleaned_df, best_pipeline


if __name__ == "__main__":
    train_and_evaluate()
    print("Training completed. Model artifacts saved to the models/ directory.")
