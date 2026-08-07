from urllib.request import urlretrieve

from src.config import FEATURES_RAW_PATH, RAW_DIR, STATEMENTS_RAW_PATH


FEATURES_URL = (
    "https://physionet.org/files/ptb-xl-plus/1.0.1/"
    "features/12sl_features.csv?download="
)

STATEMENTS_URL = (
    "https://physionet.org/files/ptb-xl-plus/1.0.1/"
    "labels/ptbxl_statements.csv?download="
)


def download_file(url, path):
    if path.exists():
        print(f"File already exists: {path.name}")
        return

    print(f"Downloading: {path.name}")
    urlretrieve(url, path)
    print(f"Saved: {path}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    download_file(FEATURES_URL, FEATURES_RAW_PATH)
    download_file(STATEMENTS_URL, STATEMENTS_RAW_PATH)


if __name__ == "__main__":
    main()