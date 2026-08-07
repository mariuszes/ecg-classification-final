from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import TRAIN_PATH
from src.utils import split_x_y


@lru_cache(maxsize=1)
def reference_profile(path=TRAIN_PATH):
    df = pd.read_csv(path)
    X, _ = split_x_y(df)
    return X.mean(), X.std().replace(0, 1)


def modify_batch(batch: pd.DataFrame, reference_std, scenario: str, strength: float) -> pd.DataFrame:
    changed = batch.copy()
    features = [col for col in reference_std.index if col in changed.columns]

    if scenario == "noisy":
        noise = np.random.default_rng(42).normal(size=changed[features].shape)
        changed[features] += noise * reference_std[features].to_numpy() * strength
    elif scenario == "drifted":
        changed[features] += reference_std[features] * strength

    return changed


def simple_drift_score(batch: pd.DataFrame, reference_mean, reference_std) -> float:
    X, _ = split_x_y(batch)
    common = [col for col in X.columns if col in reference_mean.index]
    diff = ((X[common].mean() - reference_mean[common]).abs() / reference_std[common]).mean()
    return round(float(diff), 4)
