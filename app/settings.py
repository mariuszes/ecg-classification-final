import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"
FEATURES_PATH = BASE_DIR / "models" / "production" / "features.json"

API_URL = os.getenv("FRONTEND_API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("FRONTEND_MLFLOW_URL", "http://localhost:5000")
PROMETHEUS_URL = os.getenv("FRONTEND_PROMETHEUS_URL", "http://localhost:9090")
GRAFANA_URL = os.getenv("FRONTEND_GRAFANA_URL", "http://localhost:3000")

MODEL_URI = os.getenv("MODEL_URI", "models:/ecg_classifier@champion")
