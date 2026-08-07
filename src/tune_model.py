import argparse
import random

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import BEST_PARAMS_PATH, MLFLOW_EXPERIMENT, TRAIN_PATH, TUNING_RESULTS_PATH, VAL_PATH
from src.utils import classification_metrics, save_json, split_x_y

PARAM_SPACE = {
    "n_estimators": [80, 120, 160],
    "max_depth": [None, 8, 16, 24],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"],
}


def sample_params() -> dict:
    return {key: random.choice(values) for key, values in PARAM_SPACE.items()} # dict comprehension


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--metric", default="f1")
    args = parser.parse_args()

    random.seed(42)
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    X_train, y_train = split_x_y(train_df)
    X_val, y_val = split_x_y(val_df)

    rows = []
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for trial in range(1, args.trials + 1):
        params = sample_params()
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        metrics = classification_metrics(model, X_val, y_val)

        row = {"trial": trial, **params, **metrics}
        rows.append(row)

        with mlflow.start_run(run_name=f"rf_tuning_{trial}"):
            mlflow.log_params(params)
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

        print(f"Trial {trial}: {args.metric}={metrics.get(args.metric)} params={params}")

    results = pd.DataFrame(rows).sort_values(args.metric, ascending=False) # list of dicts -> pandas dataframe
    results.to_csv(TUNING_RESULTS_PATH, index=False)

    best = results.iloc[0].to_dict()

    best_params = {}
    for key in PARAM_SPACE.keys():
        value = best[key]

        if pd.isna(value):
            value = None

        if key in ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split"] and value is not None:
            value = int(value)

        best_params[key] = value

    save_json(best_params, BEST_PARAMS_PATH)
    print(f"Best params saved: {BEST_PARAMS_PATH}")
    print(best_params)


if __name__ == "__main__":
    main()
