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
from catboost import CatBoostClassifier

HYPOTHESIS = "hyperparameters: min_data_in_leaf=20 to stabilize small Lossguide leaves"


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train a model on (X_train, y_train) and return predictions on X_val."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    feature_cols = numeric_cols + categorical_cols

    # Hold out 10% of train as early-stopping signal
    n_val = max(1, int(0.1 * len(X_train)))
    X_es = X_train.iloc[-n_val:][feature_cols]
    y_es = y_train[-n_val:]
    X_tr = X_train.iloc[:-n_val][feature_cols]
    y_tr = y_train[:-n_val]

    model = CatBoostClassifier(
        thread_count=-1,
        early_stopping_rounds=50,
        auto_class_weights="Balanced",
        grow_policy="Lossguide",
        max_leaves=63,
        rsm=0.8,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        min_data_in_leaf=20,
    )
    model.fit(
        X_tr, y_tr,
        cat_features=categorical_cols,
        eval_set=(X_es, y_es),
        verbose=False,
    )
    return model.predict_proba(X_val[feature_cols])
