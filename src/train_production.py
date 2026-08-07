import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    BEST_PARAMS_PATH,
    MLFLOW_EXPERIMENT,
    PRODUCTION_DIR,
    PRODUCTION_FEATURES_PATH,
    PRODUCTION_METRICS_PATH,
    PRODUCTION_MODEL_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from src.utils import classification_metrics, load_json, make_dirs, save_json, save_model, split_x_y


def main():
    make_dirs(PRODUCTION_DIR)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    X_train, y_train = split_x_y(train_df)
    X_val, y_val = split_x_y(val_df)

    params = load_json(BEST_PARAMS_PATH) if BEST_PARAMS_PATH.exists() else {}

    model = RandomForestClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    metrics = classification_metrics(model, X_val, y_val)

    save_model(model, PRODUCTION_MODEL_PATH)
    save_json(metrics, PRODUCTION_METRICS_PATH)
    save_json(list(X_train.columns), PRODUCTION_FEATURES_PATH)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="production_random_forest"):
        mlflow.set_tag("model_type", "production")
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({f"val_{key}": value for key, value in metrics.items()})
        mlflow.sklearn.log_model(model, name="model")

    print("Production Random Forest saved.")
    print(metrics)


if __name__ == "__main__":
    main()