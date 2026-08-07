import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import (
    CHAMPION_ALIAS, 
    MLFLOW_EXPERIMENT, 
    REGISTERED_MODEL_NAME, 
    TEST_PATH
)
from src.utils import classification_metrics, split_x_y

PRODUCTION_RUN_NAME = "production_random_forest"
MODEL_ARTIFACT_NAME = "model"
SELECTION_METRIC = "val_f1"


def main():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)

    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {MLFLOW_EXPERIMENT}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{PRODUCTION_RUN_NAME}' AND tags.model_type = 'production'",
        order_by=[f"metrics.{SELECTION_METRIC} DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No production run found. Run train_production.py first.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"

    model = mlflow.sklearn.load_model(model_uri)
    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test = split_x_y(test_df)
    test_metrics = classification_metrics(model, X_test, y_test)

    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    existing_versions = [version for version in versions if version.run_id == run_id]

    if existing_versions:
        model_version = max(existing_versions, key=lambda item: int(item.version))
        print(f"Model version already exists: {REGISTERED_MODEL_NAME} v{model_version.version}")
    else:
        model_version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        print(f"Registered new model version: {REGISTERED_MODEL_NAME} v{model_version.version}")

    for key, value in best_run.data.metrics.items():
        if key.startswith("val_"):
            client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version.version, key, str(value))

    for key, value in test_metrics.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version.version, f"test_{key}", str(value))

    client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version.version, "selected_by", SELECTION_METRIC)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version.version, "source_run_id", run_id)

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=model_version.version,
    )

    print(f"Alias: {CHAMPION_ALIAS} -> version {model_version.version}")
    print(f"Source run: {run_id}")
    print(f"Model URI: {model_uri}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()