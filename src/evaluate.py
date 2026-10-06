from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.config import EVALUATION_DIR, FIGURES_DIR


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="ckd", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="ckd", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="ckd", zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true == "ckd", y_prob))
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def save_confusion_matrix(y_true, y_pred, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["ckd", "notckd"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model, preprocessor, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    transformed_names = list(preprocessor.get_feature_names_out()) if hasattr(preprocessor, "get_feature_names_out") else None
    if transformed_names is None:
        return

    importance_values = None
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_).ravel()
    if importance_values is None:
        return

    importance = pd.DataFrame({"feature": transformed_names, "importance": importance_values})
    importance = importance.sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["feature"], importance["importance"], color="steelblue")
    ax.invert_yaxis()
    ax.set_title("Top feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics_report(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_roc_curve(y_true, y_prob, output_path: str | Path) -> None:
    from sklearn.metrics import roc_curve

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if y_prob is None:
        return

    y_binary = (y_true == "ckd").astype(int)
    fpr, tpr, _ = roc_curve(y_binary, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def report_classification(y_true, y_pred, y_prob=None) -> dict:
    metrics = compute_metrics(y_true, y_pred, y_prob)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report
    return metrics


def create_evaluation_outputs(y_true, y_pred, y_prob, model_name, output_dir: str | Path = EVALUATION_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = report_classification(y_true, y_pred, y_prob)
    save_metrics_report(metrics, output_dir / "metrics.json")
    save_confusion_matrix(y_true, y_pred, output_dir.parent / "figures" / f"{model_name}_confusion_matrix.png")
    if y_prob is not None:
        save_roc_curve(y_true, y_prob, output_dir.parent / "figures" / f"{model_name}_roc_curve.png")
    return metrics
