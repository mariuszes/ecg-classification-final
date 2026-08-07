from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "ptb-xl-plus"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = PROCESSED_DIR / "evaluation"

FEATURES_RAW_PATH = RAW_DIR / "12sl_features.csv"
STATEMENTS_RAW_PATH = RAW_DIR / "ptbxl_statements.csv"
MERGED_PATH = PROCESSED_DIR / "merged_12sl_ptbxl.csv"
CLEAN_DATA_PATH = PROCESSED_DIR / "12sl_norm_binary_clean.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
TUNING_RESULTS_PATH = PROCESSED_DIR / "tuning_results.csv"

MODELS_DIR = BASE_DIR / "models"
BASELINE_DIR = MODELS_DIR / "baseline"
PRODUCTION_DIR = MODELS_DIR / "production"

BASELINE_MODEL_PATH = BASELINE_DIR / "model.joblib"
PRODUCTION_MODEL_PATH = PRODUCTION_DIR / "model.joblib"
BASELINE_METRICS_PATH = BASELINE_DIR / "metrics.json"
PRODUCTION_METRICS_PATH = PRODUCTION_DIR / "metrics.json"
BASELINE_FEATURES_PATH = BASELINE_DIR / "features.json"
PRODUCTION_FEATURES_PATH = PRODUCTION_DIR / "features.json"
BEST_PARAMS_PATH = PRODUCTION_DIR / "best_params.json"

REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "ecg_classifier")
CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "ECG Random Forest")
