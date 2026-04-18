"""
AgenticML solution module.

This file is the only thing you (the agent) edit. The harness in ``harness/``
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

What the harness captures automatically
---------------------------------------
``mlflow.autolog()`` is enabled in the worker before ``fit_predict`` runs.
Any sklearn / xgboost / lightgbm estimator you call ``.fit()`` on has its
hyperparameters logged automatically into a per-fold nested MLflow run.
You do NOT need to import mlflow or log anything yourself.

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
from sklearn.preprocessing import StandardScaler

HYPOTHESIS = "hyperparameters: max_bin=8191"


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train a model on (X_train, y_train) and return predictions on X_val."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    classes, counts = np.unique(y_train, return_counts=True)
    balanced_weights = len(y_train) / (len(classes) * counts.astype(float))
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, balanced_weights)}
    rarest_class = int(classes[np.argmin(counts)])
    class_weight[rarest_class] *= 1.15

    X_train_model = X_train.copy()
    X_val_model = X_val.copy()

    interaction_col = "Mulching_Used__Crop_Growth_Stage"
    if {"Mulching_Used", "Crop_Growth_Stage"}.issubset(X_train_model.columns):
        X_train_model[interaction_col] = (
            X_train_model["Mulching_Used"].astype(str)
            + "__"
            + X_train_model["Crop_Growth_Stage"].astype(str)
        )
        X_val_model[interaction_col] = (
            X_val_model["Mulching_Used"].astype(str)
            + "__"
            + X_val_model["Crop_Growth_Stage"].astype(str)
        )
        categorical_cols.append(interaction_col)

    scaler = StandardScaler()
    X_train_model.loc[:, numeric_cols] = scaler.fit_transform(X_train_model[numeric_cols])
    X_val_model.loc[:, numeric_cols] = scaler.transform(X_val_model[numeric_cols])

    for col in categorical_cols:
        X_train_model[col] = X_train_model[col].astype("category")
        X_val_model[col] = pd.Categorical(
            X_val_model[col],
            categories=X_train_model[col].cat.categories,
        )

    model = LGBMClassifier(
        class_weight=class_weight,
        max_bin=8191,
        min_child_samples=175,
        n_estimators=200,
        num_leaves=24,
    )
    model.fit(X_train_model, y_train, categorical_feature=categorical_cols)
    return model.predict_proba(X_val_model)
