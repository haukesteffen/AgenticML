"""Full CV worker for OOF-backed ensemble experiments."""
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
from harness.ensemble_utils import build_meta_frames, build_predictions_manifest, log_json_artifact
from harness.metric import get_metric
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
    y_raw = train_df[cfg.dataset.target]

    if cfg.dataset.problem_type != "regression":
        y, classes = pd.factorize(y_raw, sort=True)
        n_classes = len(classes)
    else:
        y = y_raw.values.astype(float)
        classes = None
        n_classes = 0

    sys.path.insert(0, str(cfg.project_root))
    import ensemble

    meta_train, meta_test, lineage = build_meta_frames(
        cfg,
        train_df,
        test_df,
        n_classes,
        getattr(ensemble, "SOURCES", None),
        classes=classes,
    )

    metric_fn, _ = get_metric(cfg.metric.name)
    cv = build_cv(cfg.dataset.problem_type, cfg.cv)
    split_args = (meta_train, y) if cfg.dataset.problem_type != "regression" else (meta_train,)

    if cfg.dataset.problem_type == "multiclass_classification":
        oof = np.full((len(y), n_classes), np.nan)
    else:
        oof = np.full(len(y), np.nan)

    fold_scores: list[float] = []
    parent_run_id = os.environ.pop("MLFLOW_RUN_ID")
    client = mlflow.tracking.MlflowClient()
    client.set_tag(parent_run_id, "source_count", str(len(lineage)))
    log_json_artifact(client, parent_run_id, "sources.json", {"sources": lineage})
    fold_seconds = cfg.budget.fold_seconds

    prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(*split_args)):
        X_tr, y_tr = meta_train.iloc[tr_idx], y[tr_idx]
        X_va, y_va = meta_train.iloc[va_idx], y[va_idx]

        signal.alarm(fold_seconds)
        preds = ensemble.fit_predict(X_tr, y_tr, X_va)
        signal.alarm(0)

        preds = np.asarray(preds)
        validate_predictions(preds, len(va_idx), n_classes, cfg.dataset.problem_type)

        oof[va_idx] = preds
        score = metric_fn(y_va, preds, cfg.dataset.problem_type)
        client.log_metric(parent_run_id, "fold_score", score, step=fold_i)
        fold_scores.append(score)

    signal.signal(signal.SIGALRM, prev_handler)

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    client.log_metric(parent_run_id, "mean_score", mean_score)
    client.log_metric(parent_run_id, "std_score", std_score)

    test_preds = ensemble.fit_predict(meta_train, y, meta_test)
    test_preds = np.asarray(test_preds)
    validate_predictions(test_preds, len(test_df), n_classes, cfg.dataset.problem_type)

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
