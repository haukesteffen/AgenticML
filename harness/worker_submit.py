"""Refit+predict worker — fits solution.py on full train, predicts the test set.

Reads the solution module from HARNESS_SOLUTION_PATH (the downloaded artifact,
not the working-tree solution.py) and writes submission.csv + test_predictions.npy
into HARNESS_OUT_DIR.

Exit codes:
  0 = success
  1 = crash
  2 = invalid output
  3 = refit timeout
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from harness.config import HarnessConfig
from harness.worker_smoke import InvalidOutput, validate_predictions

EXIT_REFIT_TIMEOUT = 3


class RefitTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RefitTimeout


def _load_solution_module(solution_path: Path):
    spec = importlib.util.spec_from_file_location("submitted_solution", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load solution module from {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = HarnessConfig.load(args.config)

    solution_path = Path(os.environ["HARNESS_SOLUTION_PATH"])
    out_dir = Path(os.environ["HARNESS_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(cfg.project_root / cfg.dataset.train_path)
    test_df = pd.read_csv(cfg.project_root / "data" / "test.csv")
    sample_submission = pd.read_csv(cfg.project_root / "data" / "sample_submission.csv", nrows=0)

    X_train = train_df.drop(columns=[cfg.dataset.target])
    if cfg.dataset.id_column in X_train.columns:
        X_train = X_train.drop(columns=[cfg.dataset.id_column])
    y_raw = train_df[cfg.dataset.target]

    test_ids = test_df[cfg.dataset.id_column]
    X_test = test_df.drop(columns=[cfg.dataset.id_column])

    if cfg.dataset.problem_type != "regression":
        y, classes = pd.factorize(y_raw, sort=True)
        n_classes = len(classes)
    else:
        y = y_raw.values.astype(float)
        classes = None
        n_classes = 0

    solution = _load_solution_module(solution_path)

    timeout = cfg.budget.fold_seconds * cfg.cv.n_splits
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        preds = solution.fit_predict(X_train, y, X_test)
    finally:
        signal.alarm(0)

    preds = np.asarray(preds)
    validate_predictions(preds, len(X_test), n_classes, cfg.dataset.problem_type)

    if cfg.dataset.problem_type == "multiclass_classification":
        labels = classes[np.argmax(preds, axis=1)]
    elif cfg.dataset.problem_type == "binary_classification":
        if preds.ndim == 2:
            labels = classes[np.argmax(preds, axis=1)]
        else:
            labels = classes[(preds >= 0.5).astype(int)]
    else:
        labels = preds

    submission_cols = list(sample_submission.columns)
    submission = pd.DataFrame({
        submission_cols[0]: test_ids.values,
        submission_cols[1]: labels,
    })
    submission_path = out_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    np.save(str(out_dir / "test_predictions.npy"), preds)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except RefitTimeout:
        print("Killed: refit timeout exceeded", file=sys.stderr)
        sys.exit(EXIT_REFIT_TIMEOUT)
    except InvalidOutput:
        traceback.print_exc()
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(code)
