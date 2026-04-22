"""
AgenticML solution module.

This file is what you edit for base-model experiments. The harness in ``harness/``
runs cross-validation over the dataset and calls ``fit_predict`` once per fold,
passing only the training and validation slices of that fold. You never see
the other folds, the test set, or the CV indices themselves.

Contract
--------
Define exactly two things at module scope:

  HYPOTHESIS : str
      A one-line plain string literal describing what this attempt tries.
      Used as the git commit message and MLflow tag. Must be a literal — it
      is read via ast.parse without executing the module.

  fit_predict(X_train, y_train, X_val) -> np.ndarray
      Train your model on (X_train, y_train) and return predictions on X_val.

Inputs
------
  X_train : pandas.DataFrame  — training fold features (id column already dropped)
  y_train : numpy.ndarray     — training fold targets (integer-encoded for
                                 classification via pd.factorize, float for regression)
  X_val   : pandas.DataFrame  — validation fold features (id column already dropped)

Return shape
------------
  2D array of shape (len(X_val), n_classes),
  per-class probabilities with columns in
  ascending class-index order (matching pd.factorize
  with sort=True)

Rules
-----
- Do not import or call mlflow — the harness owns logging.
- Do not touch anything under ``harness/``, ``data/``, ``.env``, or ``config.yaml``.
- Do not read test data — you only have what arrives via function arguments.
- Feature engineering must be done inside ``fit_predict`` so it runs on the
  training fold only (no cross-fold leakage).
- HYPOTHESIS must be a plain string literal at module scope.
- Change exactly one axis per attempt: feature engineering, preprocessing,
  hyperparameters, or ensembling. HYPOTHESIS must be a single clause naming
  that one axis. Bundling changes hides which one moved the score and kills
  the next iteration's ability to ablate or revert. This rule is strict.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score

HYPOTHESIS = "hyperparameter: max_bin=4095 pushing histogram resolution further past the 512 win"


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train a model on (X_train, y_train) and return predictions on X_val."""
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    X_train = X_train.copy()
    X_val = X_val.copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    # Fold-internal holdout for multiplier tuning (80/20 split)
    n = len(X_train)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    split = int(0.9 * n)
    tr_idx, ho_idx = perm[:split], perm[split:]

    sub_model = LGBMClassifier(class_weight="balanced", verbose=-1, n_estimators=500, learning_rate=0.05, feature_fraction=0.8, max_bin=4095)
    sub_model.fit(
        X_train.iloc[tr_idx], y_train[tr_idx],
        categorical_feature=categorical_cols,
    )
    proba_ho = sub_model.predict_proba(X_train.iloc[ho_idx])
    y_ho = y_train[ho_idx]

    # Class 0 = High under pd.factorize sort=True (alphabetical: High < Low < Medium).
    # Sweep multiplier on High column to maximise balanced accuracy on holdout.
    best_mult, best_ba = 1.0, -1.0
    for mult in np.linspace(1.0, 6.0, 51):
        p = proba_ho.copy()
        p[:, 0] *= mult
        ba = balanced_accuracy_score(y_ho, p.argmax(axis=1))
        if ba > best_ba:
            best_ba = ba
            best_mult = mult

    # Include sub_model val predictions + seed-bag 2 full-data LGBMs → 3-way average
    probas = [sub_model.predict_proba(X_val)]
    for seed in [3, 7]:
        m = LGBMClassifier(class_weight="balanced", verbose=-1, n_estimators=500, learning_rate=0.05, random_state=seed, feature_fraction=0.8, max_bin=4095)
        m.fit(X_train, y_train, categorical_feature=categorical_cols)
        probas.append(m.predict_proba(X_val))
    proba_val = np.mean(probas, axis=0)
    proba_val[:, 0] *= best_mult
    proba_val /= proba_val.sum(axis=1, keepdims=True)
    return proba_val
