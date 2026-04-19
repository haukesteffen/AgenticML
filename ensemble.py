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

HYPOTHESIS = "equal-weight average after pruning mlp2 source"

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
    """Average the selected source predictions with equal weights."""
    if not SOURCES:
        raise ValueError("SOURCES is empty. Add at least one source run before running the harness.")

    aliases: list[str] = []
    for column in X_val.columns:
        alias = column.split("__", 1)[0]
        if alias not in aliases:
            aliases.append(alias)

    blocks = [X_val[[col for col in X_val.columns if col.startswith(f"{alias}__")]].to_numpy()
              for alias in aliases]
    preds = np.mean(blocks, axis=0)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds[:, 0]
    return preds
