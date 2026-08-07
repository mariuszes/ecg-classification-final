import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def make_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def split_x_y(df: pd.DataFrame):
    if "target" not in df.columns:
        raise ValueError("Column 'target' was not found.")
    y = df["target"].astype(int)
    X = df.drop(columns=["target", "ecg_id"], errors="ignore")
    return X, y


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def classification_metrics(model, X, y):
    pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y, proba)

    return {key: round(float(value), 4) for key, value in metrics.items()}
