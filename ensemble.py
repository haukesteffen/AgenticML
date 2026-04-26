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

HYPOTHESIS = "model: swap HGB to LightGBM stacker with class_weight=balanced + log-odds features"

SOURCES = [
    {"alias": "catboost2", "branch": "exp/catboost2", "selector": "best_improved"},
    {"alias": "lightgbm3", "branch": "exp/lightgbm3", "selector": "best_improved"},
    {"alias": "linear2", "branch": "exp/linear2", "selector": "best_improved"},
    {"alias": "mlp3", "branch": "exp/mlp3", "selector": "best_improved"},
    {"alias": "xgb2", "branch": "exp/xgb2", "selector": "best_improved"},
    {"alias": "tabm", "branch": "exp/tabm", "selector": "best_improved"},
    {"alias": "tabicl", "branch": "exp/tabicl", "selector": "best_improved"},
    {"alias": "knn", "branch": "exp/knn", "selector": "best_improved"},
    {"alias": "formula", "branch": "exp/formula", "selector": "best_improved"},
]


def _add_logodds(df: pd.DataFrame) -> np.ndarray:
    X = df.to_numpy(dtype=float)
    X_clipped = np.clip(X, 1e-7, 1 - 1e-7)
    logodds = np.log(X_clipped / (1 - X_clipped))
    return np.concatenate([X, logodds], axis=1)


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """LightGBM stacker with class_weight=balanced and raw probs + log-odds features."""
    import lightgbm as lgb

    X_tr = _add_logodds(X_train)
    X_v = _add_logodds(X_val)

    model = lgb.LGBMClassifier(
        class_weight="balanced",
        learning_rate=0.05,
        n_estimators=400,
        max_depth=5,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_train)
    return model.predict_proba(X_v)
