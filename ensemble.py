"""
AgenticML ensemble module.

This file is what you edit for OOF-backed ensemble experiments. The harness
resolves source runs from MLflow, downloads each source run's
``oof_predictions.npy`` artifact, and builds a meta-feature table for you.
Your ``fit_predict`` function sees only those meta-features, not the raw
competition features.

Contract
--------
Define exactly three things at module scope:

  HYPOTHESIS : str
      A one-line plain string literal describing what this attempt tries.
      Used as the git commit message and MLflow tag. Must be a literal.

  SOURCES : list[dict]
      Each source selects one logged run whose OOF predictions become columns
      in the meta-feature table. Supported forms are:

        {"alias": "lgbm", "branch": "exp/lightgbm", "selector": "best_improved"}
        {"alias": "xgb_best", "run_id": "<mlflow-run-id>"}

      ``alias`` is optional but strongly recommended. If omitted, the harness
      derives one from the branch or run id.

  fit_predict(X_train, y_train, X_val) -> np.ndarray
      Train an ensemble or stacker on the meta-feature table and return
      predictions for ``X_val``.

Inputs
------
  X_train : pandas.DataFrame  — source-prediction features for training rows
  y_train : numpy.ndarray     — training targets
  X_val   : pandas.DataFrame  — source-prediction features for validation rows

Meta-feature columns
--------------------
Multiclass sources contribute one column per class:
  ``<alias>__class_0``, ``<alias>__class_1``, ...

Binary-classification and regression sources contribute one column:
  ``<alias>__pred``

Rules
-----
- Do not import or call mlflow — the harness owns artifact resolution.
- Do not touch anything under ``harness/``, ``data/``, ``.env``, or ``config.yaml``.
- Use only the meta-features provided via ``X_train`` / ``X_val``.
- HYPOTHESIS must be a plain string literal at module scope.
"""

import numpy as np
import pandas as pd

HYPOTHESIS = "sources: upgrade lgbm3->lightgbm4 and xgb2->xgb3 for improved base model OOFs"

SOURCES = [
    {"alias": "catboost2", "branch": "exp/catboost2", "selector": "best_improved"},
    {"alias": "lightgbm4", "branch": "exp/lightgbm4", "selector": "best_improved"},
    {"alias": "linear2", "branch": "exp/linear2", "selector": "best_improved"},
    {"alias": "mlp3", "branch": "exp/mlp3", "selector": "best_improved"},
    {"alias": "xgb3", "branch": "exp/xgb3", "selector": "best_improved"},
    {"alias": "tabm", "branch": "exp/tabm", "selector": "best_improved"},
    {"alias": "tabicl", "branch": "exp/tabicl", "selector": "best_improved"},
    {"alias": "knn", "branch": "exp/knn", "selector": "best_improved"},
    {"alias": "formula", "branch": "exp/formula", "selector": "best_improved"},
]


def _add_logodds(X: np.ndarray) -> np.ndarray:
    X_clipped = np.clip(X, 1e-7, 1 - 1e-7)
    logodds = np.log(X_clipped / (1 - X_clipped))
    return np.concatenate([X, logodds], axis=1)


def _isotonic_calibrate(X_tr: np.ndarray, y_tr: np.ndarray, X_v: np.ndarray, n_classes: int) -> tuple:
    from sklearn.isotonic import IsotonicRegression
    X_tr_cal = np.empty_like(X_tr)
    X_v_cal = np.empty_like(X_v)
    for j in range(X_tr.shape[1]):
        class_idx = j % n_classes
        classes = np.unique(y_tr)
        y_bin = (y_tr == classes[class_idx]).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip")
        X_tr_cal[:, j] = ir.fit_transform(X_tr[:, j], y_bin)
        X_v_cal[:, j] = ir.transform(X_v[:, j])
    return X_tr_cal, X_v_cal


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Isotonic-calibrated source probs → log-odds → 5-seed CatBoost bag."""

    X_raw_tr = X_train.to_numpy(dtype=float)
    X_raw_v = X_val.to_numpy(dtype=float)

    n_classes = 3
    X_cal_tr, X_cal_v = _isotonic_calibrate(X_raw_tr, y_train, X_raw_v, n_classes)

    X_tr = _add_logodds(X_cal_tr)
    X_v = _add_logodds(X_cal_v)

    from catboost import CatBoostClassifier

    seeds = [42, 7, 13, 99, 123, 17, 31, 55, 77]
    preds = []
    for seed in seeds:
        model = CatBoostClassifier(
            auto_class_weights="Balanced",
            learning_rate=0.05,
            iterations=400,
            depth=5,
            l2_leaf_reg=3.0,
            random_seed=seed,
            verbose=0,
        )
        model.fit(X_tr, y_train)
        preds.append(model.predict_proba(X_v))
    return np.mean(preds, axis=0)
