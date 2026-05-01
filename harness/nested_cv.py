"""Nested 5x5 cross-validation that produces honest L1 OOF artifacts.

For each outer fold f:
  - The outer holdout rows get one prediction each (collected into ``outer_oof``).
  - The outer training rows get a prediction for fold-axis f via an inner
    5-fold CV (collected into ``inner_oof[:, f]``); rows in the outer holdout
    have NaN at axis f.

After all outer folds, every row has exactly one ``outer_oof`` entry. Phase e
reads ``inner_oof[:, f]`` to build L2's training data per outer fold and
``outer_oof`` to score L2 on the outer-holdout rows of each outer fold.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from harness.config import CVConfig
from harness.cv import build_cv


def nested_oof(
    fit_predict: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y: np.ndarray,
    problem_type: str,
    cv_config: CVConfig,
    n_classes: int,
    on_outer_fold_done: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run nested CV. Returns (outer_oof, inner_oof).

    Shapes:
      - regression / binary:  outer_oof (N,)        inner_oof (N, K)
      - multiclass:           outer_oof (N, C)       inner_oof (N, K, C)

    where K = cv_config.n_splits and C = n_classes. Inner-OOF entries for rows
    that were in outer fold f's holdout are NaN.
    """
    n = len(y)
    K = cv_config.n_splits
    multiclass = problem_type == "multiclass_classification"

    if multiclass:
        outer_oof = np.full((n, n_classes), np.nan)
        inner_oof = np.full((n, K, n_classes), np.nan)
    else:
        outer_oof = np.full(n, np.nan)
        inner_oof = np.full((n, K), np.nan)

    outer_cv = build_cv(problem_type, cv_config)
    outer_split_args = (X, y) if problem_type != "regression" else (X,)

    for f, (outer_tr_idx, outer_va_idx) in enumerate(outer_cv.split(*outer_split_args)):
        X_outer_tr = X.iloc[outer_tr_idx]
        y_outer_tr = y[outer_tr_idx]
        X_outer_va = X.iloc[outer_va_idx]

        inner_cv = build_cv(
            problem_type,
            n_splits=K,
            shuffle=cv_config.shuffle,
            seed=cv_config.seed + f + 1,
        )
        inner_split_args = (X_outer_tr, y_outer_tr) if problem_type != "regression" else (X_outer_tr,)

        for inner_tr_idx, inner_va_idx in inner_cv.split(*inner_split_args):
            X_inner_tr = X_outer_tr.iloc[inner_tr_idx]
            y_inner_tr = y_outer_tr[inner_tr_idx]
            X_inner_va = X_outer_tr.iloc[inner_va_idx]

            preds = np.asarray(fit_predict(X_inner_tr, y_inner_tr, X_inner_va))
            global_idx = outer_tr_idx[inner_va_idx]
            inner_oof[global_idx, f] = preds

        outer_preds = np.asarray(fit_predict(X_outer_tr, y_outer_tr, X_outer_va))
        outer_oof[outer_va_idx] = outer_preds

        if on_outer_fold_done is not None:
            on_outer_fold_done(f, outer_va_idx, outer_preds)

    return outer_oof, inner_oof
