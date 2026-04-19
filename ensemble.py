"""
AgenticML ensemble module.

This file is what you edit for OOF-backed ensemble experiments. The harness
resolves source runs from MLflow, downloads each source run's ``oof.npy``
artifact, and builds a meta-feature table for you. Your ``fit_predict``
function sees only those meta-features, not the raw competition features.

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

HYPOTHESIS = "holdout-tuned class-prior calibration for logistic stacker"

SOURCES = [
    {"alias": "catboost", "branch": "exp/catboost", "selector": "best_improved"},
    {"alias": "lightgbm2", "branch": "exp/lightgbm2", "selector": "best_improved"},
    {"alias": "xgb", "branch": "exp/xgb", "selector": "best_improved"},
]


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Fit logistic stacking, then class-prior calibrate probabilities."""
    if not SOURCES:
        raise ValueError("SOURCES is empty. Add at least one source run before running the harness.")

    X_train_np = X_train.to_numpy()
    X_val_np = X_val.to_numpy()
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_np,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    calib_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    calib_model.fit(X_fit, y_fit)
    cal_proba = calib_model.predict_proba(X_cal)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_train_np, y_train)
    val_proba = model.predict_proba(X_val_np)
    if val_proba.ndim == 2 and val_proba.shape[1] == 1:
        return val_proba[:, 0]

    class_counts = np.bincount(y_train, minlength=val_proba.shape[1]).astype(float)
    class_priors = np.clip(class_counts / class_counts.sum(), 1e-6, 1.0)

    best_gamma = 0.0
    best_score = -np.inf
    for gamma in np.linspace(0.0, 1.5, 16):
        adjusted = cal_proba / np.power(class_priors, gamma)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        pred_labels = adjusted.argmax(axis=1)
        score = balanced_accuracy_score(y_cal, pred_labels)
        if score > best_score:
            best_score = score
            best_gamma = float(gamma)

    val_adjusted = val_proba / np.power(class_priors, best_gamma)
    val_adjusted /= val_adjusted.sum(axis=1, keepdims=True)
    return val_adjusted
