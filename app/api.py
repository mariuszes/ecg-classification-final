import json
import time
from functools import lru_cache
from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from app.settings import FEATURES_PATH, MODEL_URI, TEST_PATH

app = FastAPI(title="ECG Monitoring API")

prediction_requests = Counter(
    "ecg_prediction_requests_total",
    "Number of prediction requests",
    ["scenario"],
)
prediction_errors = Counter(
    "ecg_prediction_errors_total",
    "Number of prediction errors",
    ["scenario"],
)
prediction_latency = Histogram(
    "ecg_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["scenario"],
)
prediction_classes = Counter(
    "ecg_prediction_class_total",
    "Number of predictions by class",
    ["scenario", "predicted_class"],
)
feature_drift_score = Gauge(
    "ecg_feature_drift_score",
    "Last feature drift score for a simulation batch",
    ["scenario"],
)

for scenario in ["normal", "noisy", "drifted"]:
    prediction_errors.labels(scenario=scenario).inc(0)

app.mount("/metrics", make_asgi_app())

Scenario = Literal["normal", "noisy", "drifted"]

# /predict-row
class RowRequest(BaseModel):
    row_index: int

# /predict 
class FeatureRequest(BaseModel):
    features: dict[str, float]
    scenario: Scenario = "normal"

# /simulation-metrics
class SimulationMetricsRequest(BaseModel):
    scenario: Scenario
    drift_score: float


@lru_cache(maxsize=1)
def get_model():
    return mlflow.sklearn.load_model(MODEL_URI)


@lru_cache(maxsize=1)
def get_features():
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, encoding="utf-8") as file:
            return json.load(file)
    df = pd.read_csv(TEST_PATH, nrows=1)
    return [col for col in df.columns if col not in ["target", "ecg_id"]]


def get_test_row(row_index: int):
    df = pd.read_csv(TEST_PATH)
    if row_index < 0 or row_index >= len(df):
        raise HTTPException(status_code=400, detail="row_index is outside test.csv range")
    row = df.iloc[row_index]
    features = get_features()
    X = pd.DataFrame([row[features].to_dict()])
    true_target = int(row["target"]) if "target" in row else None
    ecg_id = int(row["ecg_id"]) if "ecg_id" in row else row_index
    return ecg_id, true_target, X


def make_prediction(X: pd.DataFrame, scenario: str):
    start = time.perf_counter()
    prediction_requests.labels(scenario=scenario).inc()

    try:
        model = get_model()
        pred = int(model.predict(X)[0])
        label = "abnormal" if pred == 1 else "normal"
        proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

        prediction_classes.labels(scenario=scenario, predicted_class=label).inc()
        return pred, label, proba
    except Exception:
        prediction_errors.labels(scenario=scenario).inc()
        raise
    finally:
        prediction_latency.labels(scenario=scenario).observe(time.perf_counter() - start)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        model = get_model()
        return {
            "model_uri": MODEL_URI,
            "model_type": type(model).__name__,
            "feature_count": len(get_features()),
        }
    except Exception as exc:
        raise RuntimeError(f"Model is not available: {exc}")


@app.post("/predict-row")
def predict_row(request: RowRequest):
    try:
        ecg_id, true_target, X = get_test_row(request.row_index)
        pred, label, proba = make_prediction(X, "normal")
        return {
            "row_index": request.row_index,
            "ecg_id": ecg_id,
            "prediction": pred,
            "prediction_label": label,
            "true_target": true_target,
            "true_label": "abnormal" if true_target == 1 else "normal",
            "probability_abnormal": proba,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict")
def predict(request: FeatureRequest):
    features = get_features()
    missing = [feature for feature in features if feature not in request.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing {len(missing)} model features")

    try:
        X = pd.DataFrame([[request.features[feature] for feature in features]], columns=features)
        pred, label, proba = make_prediction(X, request.scenario)
        return {
            "prediction": pred,
            "prediction_label": label,
            "probability_abnormal": proba,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/simulation-metrics")
def simulation_metrics(request: SimulationMetricsRequest):
    feature_drift_score.labels(scenario=request.scenario).set(request.drift_score)
    return {"scenario": request.scenario, "drift_score": request.drift_score}