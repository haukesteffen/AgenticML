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

HYPOTHESIS = "tune C=0.1 (stronger regularization) on logit-transformed logistic regression stacker"

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


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Raw OOF probs → logistic regression stacker."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    eps = 1e-6
    X_tr = np.clip(X_train.to_numpy(dtype=float), eps, 1 - eps)
    X_v = np.clip(X_val.to_numpy(dtype=float), eps, 1 - eps)
    X_tr = np.log(X_tr / (1 - X_tr))
    X_v = np.log(X_v / (1 - X_v))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_v = scaler.transform(X_v)

    model = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")
    model.fit(X_tr, y_train)
    return model.predict_proba(X_v)
