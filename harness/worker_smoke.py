"""Smoke-test worker — runs fit_predict on a small data subset to catch obvious bugs.

No MLflow imports. Exit codes:
  0 = success
  1 = crash (uncaught exception, traceback on stderr)
  2 = invalid output (wrong shape, NaN, etc.)
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from harness.config import HarnessConfig
from harness.cv import build_cv


class InvalidOutput(Exception):
    pass


def validate_predictions(preds: np.ndarray, n_val: int, n_classes: int, problem_type: str) -> None:
    preds = np.asarray(preds)
    if problem_type == "multiclass_classification":
        if preds.ndim != 2:
            raise InvalidOutput(f"Expected 2D predictions for multiclass, got {preds.ndim}D")
        if preds.shape != (n_val, n_classes):
            raise InvalidOutput(f"Expected shape ({n_val}, {n_classes}), got {preds.shape}")
    else:
        if preds.ndim != 1:
            raise InvalidOutput(f"Expected 1D predictions, got {preds.ndim}D")
        if preds.shape[0] != n_val:
            raise InvalidOutput(f"Expected {n_val} predictions, got {preds.shape[0]}")

    if np.any(np.isnan(preds)):
        raise InvalidOutput("Predictions contain NaN")
    if np.any(np.isinf(preds)):
        raise InvalidOutput("Predictions contain inf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = HarnessConfig.load(args.config)
    train_df = pd.read_csv(cfg.project_root / cfg.dataset.train_path)
    X = train_df.drop(columns=[cfg.dataset.target])
    if cfg.dataset.id_column in X.columns:
        X = X.drop(columns=[cfg.dataset.id_column])
    y_raw = train_df[cfg.dataset.target]

    if cfg.dataset.problem_type != "regression":
        y, uniques = pd.factorize(y_raw, sort=True)
        n_classes = len(uniques)
    else:
        y = y_raw.values.astype(float)
        n_classes = 0

    stratify = y if cfg.dataset.problem_type != "regression" else None
    if cfg.smoke.data_fraction < 1.0:
        X, _, y, _ = train_test_split(
            X, y,
            train_size=cfg.smoke.data_fraction,
            stratify=stratify,
            random_state=cfg.cv.seed,
        )
        if stratify is not None:
            stratify = y

    sys.path.insert(0, str(cfg.project_root))
    import solution

    cv = build_cv(cfg.dataset.problem_type, n_splits=cfg.smoke.n_splits, shuffle=True, seed=cfg.cv.seed)
    split_args = (X, y) if cfg.dataset.problem_type != "regression" else (X,)

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(*split_args)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va = X.iloc[va_idx]

        preds = solution.fit_predict(X_tr, y_tr, X_va)
        preds = np.asarray(preds)
        validate_predictions(preds, len(va_idx), n_classes, cfg.dataset.problem_type)

    return 0


if __name__ == "__main__":
    try:
        code = main()
    except InvalidOutput:
        traceback.print_exc()
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(code)
