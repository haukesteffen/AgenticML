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
from harness.ensemble_utils import build_meta_frame, log_json_artifact
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
    import ensemble

    meta_df, lineage = build_meta_frame(
        cfg,
        train_df,
        test_df,
        n_classes,
        getattr(ensemble, "SOURCES", None),
        classes=classes,
    )

    setup_autolog()

    metric_fn, _ = get_metric(cfg.metric.name)
    cv = build_cv(cfg.dataset.problem_type, cfg.cv)
    split_args = (meta_df, y) if cfg.dataset.problem_type != "regression" else (meta_df,)

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
        X_tr, y_tr = meta_df.iloc[tr_idx], y[tr_idx]
        X_va, y_va = meta_df.iloc[va_idx], y[va_idx]

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"fold_{fold_i}",
            tags={"mlflow.parentRunId": parent_run_id},
        ):
            signal.alarm(fold_seconds)
            preds = ensemble.fit_predict(X_tr, y_tr, X_va)
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

    with tempfile.TemporaryDirectory() as tmpdir:
        oof_path = Path(tmpdir) / "oof_predictions.npy"
        np.save(str(oof_path), oof)
        client.log_artifact(parent_run_id, str(oof_path))

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
