import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CLEAN_DATA_PATH, PROCESSED_DIR, TEST_PATH, TRAIN_PATH, VAL_PATH
from src.utils import make_dirs


def main():
    df = pd.read_csv(CLEAN_DATA_PATH)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["target"],
        random_state=42,
    )

    make_dirs(PROCESSED_DIR)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Train: {train_df.shape} -> {TRAIN_PATH}")
    print(f"Val:   {val_df.shape} -> {VAL_PATH}")
    print(f"Test:  {test_df.shape} -> {TEST_PATH}")


if __name__ == "__main__":
    main()
