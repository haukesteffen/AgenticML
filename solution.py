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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, SplineTransformer

HYPOTHESIS = "preprocessing: replace StandardScaler with RobustScaler (median+IQR) for outlier-robust numeric scaling"


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train a model on (X_train, y_train) and return predictions on X_val."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    scaler = RobustScaler()
    X_train_num = scaler.fit_transform(X_train[numeric_cols])
    X_val_num = scaler.transform(X_val[numeric_cols])

    X_train_aug = X_train.copy()
    X_val_aug = X_val.copy()
    for i, col in enumerate(numeric_cols):
        X_train_aug[col] = X_train_num[:, i]
        X_val_aug[col] = X_val_num[:, i]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        (
            "num_spline",
            SplineTransformer(n_knots=33, degree=3, include_bias=False),
            numeric_cols,
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced", C=1.5, solver="newton-cholesky")),
    ])
    pipe.fit(X_train_aug, y_train)
    return pipe.predict_proba(X_val_aug)
