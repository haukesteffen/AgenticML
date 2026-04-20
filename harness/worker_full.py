"""Full CV worker — runs all folds with MLflow autolog and per-fold child runs.

Uses MlflowClient to log aggregates to the parent run (avoids cross-process
run reactivation which is unreliable). Fold runs declare their parent via
the mlflow.parentRunId tag.

Exit codes:
  0 = success
  1 = crash
  2 = invalid output
  3 = per-fold timeout
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from harness.config import HarnessConfig
from harness.cv import build_cv
from harness.ensemble_utils import build_predictions_manifest, log_json_artifact
from harness.metric import get_metric
from harness.mlflow_utils import ensure_experiment, setup_autolog
from harness.worker_smoke import InvalidOutput, validate_predictions

EXIT_FOLD_TIMEOUT = 3


class FoldTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise FoldTimeout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = HarnessConfig.load(args.config)
    train_df = pd.read_csv(cfg.project_root / cfg.dataset.train_path)
    test_df = pd.read_csv(cfg.project_root / cfg.dataset.test_path)
    X = train_df.drop(columns=[cfg.dataset.target])
    if cfg.dataset.id_column in X.columns:
        X = X.drop(columns=[cfg.dataset.id_column])
    X_test = test_df.drop(columns=[cfg.dataset.id_column])
    y_raw = train_df[cfg.dataset.target]

    if cfg.dataset.problem_type != "regression":
        y, classes = pd.factorize(y_raw, sort=True)
        n_classes = len(classes)
    else:
        y = y_raw.values.astype(float)
        classes = None
        n_classes = 0

    branch = os.environ.get("HARNESS_BRANCH", "unknown")
    slug = os.environ.get("HARNESS_SLUG", "")
    experiment_id = ensure_experiment(f"{cfg.mlflow.experiment_prefix}_{slug}_{branch}")

    sys.path.insert(0, str(cfg.project_root))
    import solution

    setup_autolog()

    metric_fn, _ = get_metric(cfg.metric.name)
    cv = build_cv(cfg.dataset.problem_type, cfg.cv)
    split_args = (X, y) if cfg.dataset.problem_type != "regression" else (X,)

    if cfg.dataset.problem_type == "multiclass_classification":
        oof = np.full((len(y), n_classes), np.nan)
    else:
        oof = np.full(len(y), np.nan)

    fold_scores: list[float] = []
    parent_run_id = os.environ.pop("MLFLOW_RUN_ID")
    client = mlflow.tracking.MlflowClient()
    fold_seconds = cfg.budget.fold_seconds

    prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(*split_args)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"fold_{fold_i}",
            tags={"mlflow.parentRunId": parent_run_id},
        ):
            signal.alarm(fold_seconds)
            preds = solution.fit_predict(X_tr, y_tr, X_va)
            signal.alarm(0)

            preds = np.asarray(preds)
            validate_predictions(preds, len(va_idx), n_classes, cfg.dataset.problem_type)

            oof[va_idx] = preds
            score = metric_fn(y_va, preds, cfg.dataset.problem_type)
            mlflow.log_metric("fold_score", score)
            fold_scores.append(score)

    signal.signal(signal.SIGALRM, prev_handler)

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    client.log_metric(parent_run_id, "mean_score", mean_score)
    client.log_metric(parent_run_id, "std_score", std_score)

    test_preds = solution.fit_predict(X, y, X_test)
    test_preds = np.asarray(test_preds)
    validate_predictions(test_preds, len(X_test), n_classes, cfg.dataset.problem_type)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        oof_path = tmpdir_path / "oof_predictions.npy"
        np.save(str(oof_path), oof)
        client.log_artifact(parent_run_id, str(oof_path))

        test_path = tmpdir_path / "test_predictions.npy"
        np.save(str(test_path), test_preds)
        client.log_artifact(parent_run_id, str(test_path))

    manifest = build_predictions_manifest(
        cfg,
        train_df,
        test_df,
        n_classes,
        oof.shape,
        test_preds.shape,
        classes=classes,
    )
    log_json_artifact(client, parent_run_id, "predictions_manifest.json", manifest)

    return 0


if __name__ == "__main__":
    try:
        code = main()
    except FoldTimeout:
        print("Killed: per-fold timeout exceeded", file=sys.stderr)
        sys.exit(EXIT_FOLD_TIMEOUT)
    except InvalidOutput:
        traceback.print_exc()
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(code)
