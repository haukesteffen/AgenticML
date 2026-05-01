"""Smoke worker for phase e — runs ``ensemble.fit_predict`` on a tiny slice
of the resolved meta-features to catch obvious bugs before nested CV starts.
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import pandas as pd

from harness.config import HarnessConfig
from harness.promoted_resolver import resolve_sources
from harness.worker_ensemble_full import (
    _stack_from_inner,
    _stack_from_outer,
    _to_frame,
)
from harness.worker_smoke import InvalidOutput, validate_predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = HarnessConfig.load(args.config)
    train_df = pd.read_csv(cfg.project_root / cfg.dataset.train_path)
    y_raw = train_df[cfg.dataset.target]

    if cfg.dataset.problem_type != "regression":
        y, classes = pd.factorize(y_raw, sort=True)
        n_classes = len(classes)
    else:
        y = y_raw.values.astype(float)
        n_classes = 0

    sys.path.insert(0, str(cfg.project_root))
    import ensemble

    raw_sources = getattr(ensemble, "SOURCES", None)
    if not isinstance(raw_sources, list) or not all(isinstance(s, str) for s in raw_sources):
        raise TypeError("ensemble.SOURCES must be a list[str] of promoted lane names.")
    sources = resolve_sources(cfg, raw_sources)

    n = len(y)
    sample = max(64, int(n * cfg.smoke.data_fraction))
    rng = np.random.default_rng(cfg.cv.seed)
    tr_idx = rng.choice(n, size=sample, replace=False)
    va_idx = rng.choice(n, size=max(32, sample // 4), replace=False)

    X_tr_arr = _stack_from_inner(sources, tr_idx, fold=0, problem_type=cfg.dataset.problem_type)
    X_va_arr = _stack_from_outer(sources, va_idx, cfg.dataset.problem_type)

    if np.any(np.isnan(X_tr_arr)):
        # rows where fold-0 inner OOF was the outer-holdout slot get NaN; skip them
        good = ~np.any(np.isnan(X_tr_arr), axis=1)
        X_tr_arr = X_tr_arr[good]
        tr_idx = tr_idx[good]
    y_tr = y[tr_idx]

    X_tr = _to_frame(sources, X_tr_arr, n_classes, cfg.dataset.problem_type)
    X_va = _to_frame(sources, X_va_arr, n_classes, cfg.dataset.problem_type)

    preds = ensemble.fit_predict(X_tr, y_tr, X_va)
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
