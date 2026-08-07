import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    BASELINE_DIR,
    BASELINE_FEATURES_PATH,
    BASELINE_METRICS_PATH,
    BASELINE_MODEL_PATH,
    MLFLOW_EXPERIMENT,
    TRAIN_PATH,
    VAL_PATH,
)
from src.utils import classification_metrics, make_dirs, save_json, save_model, split_x_y


def main():
    make_dirs(BASELINE_DIR)
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    X_train, y_train = split_x_y(train_df)
    X_val, y_val = split_x_y(val_df)

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = classification_metrics(model, X_val, y_val)
    save_model(model, BASELINE_MODEL_PATH)
    save_json(metrics, BASELINE_METRICS_PATH)
    save_json(list(X_train.columns), BASELINE_FEATURES_PATH)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="baseline_random_forest"):
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({f"val_{i}": j for i, j in metrics.items()})
        mlflow.sklearn.log_model(model, name="model")

    print("Baseline Random Forest saved.")
    print(metrics)


if __name__ == "__main__":
    main()
