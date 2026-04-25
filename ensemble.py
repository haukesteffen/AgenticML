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

HYPOTHESIS = "Bagged LR C=0.05 (50 bootstrap) + log-odds + gamma on holdout"

SOURCES = [
    {"alias": "lgbm3", "branch": "exp/lightgbm3", "selector": "best_improved"},
    {"alias": "xgb2", "branch": "exp/xgb2", "selector": "best_improved"},
    {"alias": "catboost2", "branch": "exp/catboost2", "selector": "best_improved"},
    {"alias": "mlp3", "branch": "exp/mlp3", "selector": "best_improved"},
    {"alias": "linear2", "branch": "exp/linear2", "selector": "best_improved"},
]


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample
    from scipy.optimize import minimize

    def log_odds(X):
        p = np.clip(X.to_numpy(), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def apply_gamma(proba, gamma):
        p = np.clip(proba, 1e-10, 1)
        scaled = p ** gamma
        return scaled / scaled.sum(axis=1, keepdims=True)

    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    # Gamma calibration on holdout
    clf_ho = LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000, solver="lbfgs")
    clf_ho.fit(log_odds(X_tr), y_tr)
    proba_ho = clf_ho.predict_proba(log_odds(X_ho))
    n_classes = proba_ho.shape[1]

    def neg_ba(gamma):
        adj = apply_gamma(proba_ho, gamma)
        return -balanced_accuracy_score(y_ho, np.argmax(adj, axis=1))

    res = minimize(neg_ba, np.ones(n_classes), method="Nelder-Mead",
                   options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-7})
    gamma_opt = np.clip(res.x, 0.1, 10.0)

    # Bagged LR: 20 bootstrap models
    X_lo = log_odds(X_train)
    X_val_lo = log_odds(X_val)
    all_probas = []
    for i in range(50):
        X_b, y_b = resample(X_lo, y_train, random_state=i, stratify=y_train)
        clf_b = LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000, solver="lbfgs")
        clf_b.fit(X_b, y_b)
        all_probas.append(clf_b.predict_proba(X_val_lo))
    proba_val = np.mean(all_probas, axis=0)
    return apply_gamma(proba_val, gamma_opt)
