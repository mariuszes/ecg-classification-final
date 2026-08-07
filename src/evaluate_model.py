import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import (
    BASELINE_MODEL_PATH,
    EVALUATION_DIR,
    PRODUCTION_MODEL_PATH,
    TEST_PATH,
)
from src.utils import classification_metrics, load_model, make_dirs, split_x_y


def evaluate_one(name: str, model_path, X_test, y_test) -> dict:
    model = load_model(model_path)
    metrics = classification_metrics(model, X_test, y_test)
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )

    cm_path = EVALUATION_DIR / f"confusion_matrix_{name}.csv"
    cm_df.to_csv(cm_path)

    return {"model": name, **metrics}


def main():
    make_dirs(EVALUATION_DIR)

    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test = split_x_y(test_df)

    rows = []

    if BASELINE_MODEL_PATH.exists():
        rows.append(evaluate_one("baseline", BASELINE_MODEL_PATH, X_test, y_test))

    if PRODUCTION_MODEL_PATH.exists():
        rows.append(evaluate_one("production", PRODUCTION_MODEL_PATH, X_test, y_test))

    summary = pd.DataFrame(rows)
    output_path = EVALUATION_DIR / "evaluation_summary.csv"
    summary.to_csv(output_path, index=False)
    print(summary)
    print(f"Saved evaluation summary: {output_path}")


if __name__ == "__main__":
    main()