import ast

import pandas as pd

from src.config import CLEAN_DATA_PATH, FEATURES_RAW_PATH, MERGED_PATH, PROCESSED_DIR, STATEMENTS_RAW_PATH
from src.utils import make_dirs


def get_norm_score(text) -> float:
    try:
        codes = ast.literal_eval(text)
    except Exception:
        return 0.0

    if isinstance(codes, dict):
        return float(codes.get("NORM", 0.0))

    if isinstance(codes, list):
        for item in codes:
            if len(item) >= 2 and item[0] == "NORM":
                return float(item[1])

    return 0.0


def load_or_make_merged() -> pd.DataFrame:
    if MERGED_PATH.exists():
        return pd.read_csv(MERGED_PATH)

    if not FEATURES_RAW_PATH.exists() or not STATEMENTS_RAW_PATH.exists():
        raise FileNotFoundError("Missing raw PTB-XL+ files in data/ptb-xl-plus.")

    features = pd.read_csv(FEATURES_RAW_PATH)
    statements = pd.read_csv(STATEMENTS_RAW_PATH)
    merged = features.merge(statements, on="ecg_id", how="inner")
    merged.to_csv(MERGED_PATH, index=False)
    return merged


def main():
    make_dirs(PROCESSED_DIR)
    df = load_or_make_merged()

    df["norm_score"] = df["scp_codes"].apply(get_norm_score)
    df["target"] = df["norm_score"].apply(lambda value: 0 if value >= 80 else 1)

    X = df.drop(
        columns=["target", "scp_codes", "scp_codes_ext", "scp_codes_ext_snomed", "norm_score"],
        errors="ignore",
    )
    ecg_id = X["ecg_id"]
    X = X.drop(columns=["ecg_id"], errors="ignore")

    missing_ratio = X.isna().mean()
    columns_to_keep = missing_ratio[missing_ratio <= 0.05].index
    X = X[columns_to_keep]

    X = X.fillna(X.median(numeric_only=True))

    clean = X.copy()
    clean.insert(0, "ecg_id", ecg_id.values)
    clean["target"] = df["target"]
    clean.to_csv(CLEAN_DATA_PATH, index=False)

    print(f"Saved clean dataset: {CLEAN_DATA_PATH}")
    print(f"Shape: {clean.shape}")
    print(clean["target"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
