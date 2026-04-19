"""Smoke-test worker for OOF-backed ensemble experiments."""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from harness.config import HarnessConfig
from harness.cv import build_cv
from harness.ensemble_utils import build_meta_frame
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
        classes = None
        n_classes = 0

    sys.path.insert(0, str(cfg.project_root))
    import ensemble

    meta_df, _ = build_meta_frame(
        cfg,
        train_df,
        n_classes,
        getattr(ensemble, "SOURCES", None),
        classes=classes,
    )

    stratify = y if cfg.dataset.problem_type != "regression" else None
    if cfg.smoke.data_fraction < 1.0:
        meta_df, _, y, _ = train_test_split(
            meta_df,
            y,
            train_size=cfg.smoke.data_fraction,
            stratify=stratify,
            random_state=cfg.cv.seed,
        )
        if stratify is not None:
            stratify = y

    cv = build_cv(cfg.dataset.problem_type, n_splits=cfg.smoke.n_splits, shuffle=True, seed=cfg.cv.seed)
    split_args = (meta_df, y) if cfg.dataset.problem_type != "regression" else (meta_df,)

    for tr_idx, va_idx in cv.split(*split_args):
        X_tr, y_tr = meta_df.iloc[tr_idx], y[tr_idx]
        X_va = meta_df.iloc[va_idx]

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
