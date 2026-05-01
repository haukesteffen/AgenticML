"""Phase e: L2 nested CV worker.

Reads ``ensemble.SOURCES`` (list of lane names), resolves each lane's promoted
artifacts (``oof_outer``, ``oof_inner``, ``test_pred``), and runs L2 nested CV:

  for outer fold f:
    L2-train features = horizontally stacked source.inner_oof[:, f] for the
                        outer-train rows
    L2-val   features = horizontally stacked source.outer_oof for the
                        outer-holdout rows
    ensemble.fit_predict(...) trains one L2 per outer fold and predicts the
    outer-holdout rows -> L2 outer-OOF for those rows

After the loop, refits L2 on the full L1 outer-OOF table and predicts the test
features (horizontally stacked source.test_pred). The L2 outer-OOF and test
prediction are logged as ``oof_predictions.npy`` / ``test_predictions.npy`` so
the existing submission flow keeps working.
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
from harness.promoted_resolver import PromotedSource, resolve_sources
from harness.worker_smoke import InvalidOutput, validate_predictions

EXIT_FOLD_TIMEOUT = 3


class FoldTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise FoldTimeout


def _source_columns(lane: str, n_classes: int, problem_type: str) -> list[str]:
    if problem_type == "multiclass_classification":
        return [f"{lane}__class_{i}" for i in range(n_classes)]
    return [f"{lane}__pred"]


def _slice_block(arr: np.ndarray, idx: np.ndarray, problem_type: str) -> np.ndarray:
    """Pick rows from a 1D/2D source array as a 2D block (cols=1 for non-multiclass)."""
    block = arr[idx]
    if problem_type != "multiclass_classification" and block.ndim == 1:
        block = block[:, None]
    return block


def _stack_from_outer(sources: list[PromotedSource], idx: np.ndarray, problem_type: str) -> np.ndarray:
    return np.concatenate(
        [_slice_block(s.outer_oof, idx, problem_type) for s in sources], axis=1
    )


def _stack_from_inner(
    sources: list[PromotedSource],
    idx: np.ndarray,
    fold: int,
    problem_type: str,
) -> np.ndarray:
    blocks = []
    for s in sources:
        if s.inner_oof.ndim == 3:
            block = s.inner_oof[idx, fold, :]
        else:
            block = s.inner_oof[idx, fold][:, None]
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def _stack_from_test(sources: list[PromotedSource], problem_type: str) -> np.ndarray:
    blocks = []
    for s in sources:
        arr = s.test_pred
        if problem_type != "multiclass_classification" and arr.ndim == 1:
            arr = arr[:, None]
        blocks.append(arr)
    return np.concatenate(blocks, axis=1)


def _to_frame(
    sources: list[PromotedSource],
    matrix: np.ndarray,
    n_classes: int,
    problem_type: str,
) -> pd.DataFrame:
    cols: list[str] = []
    for s in sources:
        cols.extend(_source_columns(s.lane, n_classes, problem_type))
    return pd.DataFrame(matrix, columns=cols)


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

    raw_sources = getattr(ensemble, "SOURCES", None)
    if not isinstance(raw_sources, list) or not all(isinstance(s, str) for s in raw_sources):
        raise TypeError("ensemble.SOURCES must be a list[str] of promoted lane names.")
    sources = resolve_sources(cfg, raw_sources)

    metric_fn, _ = get_metric(cfg.metric.name)
    cv = build_cv(cfg.dataset.problem_type, cfg.cv)
    split_args = (np.zeros(len(y)), y) if cfg.dataset.problem_type != "regression" else (np.zeros(len(y)),)

    if cfg.dataset.problem_type == "multiclass_classification":
        l2_outer = np.full((len(y), n_classes), np.nan)
    else:
        l2_outer = np.full(len(y), np.nan)

    parent_run_id = os.environ.pop("MLFLOW_RUN_ID")
    client = mlflow.tracking.MlflowClient()

    lineage = [
        {
            "lane": s.lane,
            "run_id": s.run_id,
            "recipe": s.recipe,
            "family": s.family,
            "source_branch": s.source_branch,
            "source_commit": s.source_commit,
            "cv_score_outer": s.cv_score_outer,
        }
        for s in sources
    ]
    client.set_tag(parent_run_id, "source_count", str(len(sources)))
    log_json_artifact(client, parent_run_id, "sources.json", {"sources": lineage})

    fold_seconds = cfg.budget.fold_seconds
    prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    fold_scores: list[float] = []
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(*split_args)):
        X_tr_arr = _stack_from_inner(sources, tr_idx, fold_i, cfg.dataset.problem_type)
        X_va_arr = _stack_from_outer(sources, va_idx, cfg.dataset.problem_type)
        X_tr = _to_frame(sources, X_tr_arr, n_classes, cfg.dataset.problem_type)
        X_va = _to_frame(sources, X_va_arr, n_classes, cfg.dataset.problem_type)
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        signal.alarm(fold_seconds)
        preds = ensemble.fit_predict(X_tr, y_tr, X_va)
        signal.alarm(0)

        preds = np.asarray(preds)
        validate_predictions(preds, len(va_idx), n_classes, cfg.dataset.problem_type)
        l2_outer[va_idx] = preds

        score = metric_fn(y_va, preds, cfg.dataset.problem_type)
        client.log_metric(parent_run_id, f"fold_score_{fold_i}", score)
        fold_scores.append(score)

    signal.signal(signal.SIGALRM, prev_handler)

    mean_score = float(metric_fn(y, l2_outer, cfg.dataset.problem_type))
    std_score = float(np.std(fold_scores))
    client.log_metric(parent_run_id, "mean_score", mean_score)
    client.log_metric(parent_run_id, "std_score", std_score)

    X_full_arr = _stack_from_outer(sources, np.arange(len(y)), cfg.dataset.problem_type)
    X_test_arr = _stack_from_test(sources, cfg.dataset.problem_type)
    X_full = _to_frame(sources, X_full_arr, n_classes, cfg.dataset.problem_type)
    X_test = _to_frame(sources, X_test_arr, n_classes, cfg.dataset.problem_type)

    test_preds = np.asarray(ensemble.fit_predict(X_full, y, X_test))
    validate_predictions(test_preds, len(X_test), n_classes, cfg.dataset.problem_type)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        np.save(tmp / "oof_predictions.npy", l2_outer)
        np.save(tmp / "test_predictions.npy", test_preds)
        for f in ("oof_predictions.npy", "test_predictions.npy"):
            client.log_artifact(parent_run_id, str(tmp / f))

    manifest = build_predictions_manifest(
        cfg, train_df, test_df, n_classes, l2_outer.shape, test_preds.shape, classes=classes,
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
