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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HYPOTHESIS = "logit-transform OOF probs before StandardScaler: spreads clustered probabilities for better MLP input"

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
    from scipy.special import logit

    X_tr_raw = np.clip(X_train.to_numpy(), 1e-6, 1 - 1e-6)
    X_v_raw = np.clip(X_val.to_numpy(), 1e-6, 1 - 1e-6)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(logit(X_tr_raw))
    X_v = scaler.transform(logit(X_v_raw))

    all_preds = []
    for seed in [0, 1, 2]:
        model = MLPClassifier(hidden_layer_sizes=(64,), random_state=seed)
        model.fit(X_tr, y_train)
        all_preds.append(model.predict_proba(X_v))
    preds = np.mean(all_preds, axis=0)

    # inverse-frequency weight all classes, scaled so minority gets 10x
    classes, counts = np.unique(y_train, return_counts=True)
    freqs = counts / len(y_train)
    inv_freq = 1.0 / freqs
    scale = inv_freq / inv_freq.min() * 10.0
    class_order = model.classes_
    weights = np.array([scale[np.where(classes == c)[0][0]] for c in class_order])
    preds *= weights[np.newaxis, :]
    preds /= preds.sum(axis=1, keepdims=True)
    return preds
