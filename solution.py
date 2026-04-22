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
from sklearn.model_selection import train_test_split

HYPOTHESIS = "feature engineering: mean(Rainfall_mm) by Region as numeric feature for regional water context"


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train a model on (X_train, y_train) and return predictions on X_val."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    feature_cols = numeric_cols + categorical_cols

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train["Season_Stage"] = X_train["Season"].astype(str) + "_" + X_train["Crop_Growth_Stage"].astype(str)
    X_val["Season_Stage"] = X_val["Season"].astype(str) + "_" + X_val["Crop_Growth_Stage"].astype(str)
    X_train["Crop_Stage"] = X_train["Crop_Type"].astype(str) + "_" + X_train["Crop_Growth_Stage"].astype(str)
    X_val["Crop_Stage"] = X_val["Crop_Type"].astype(str) + "_" + X_val["Crop_Growth_Stage"].astype(str)

    idx_tr, idx_es = train_test_split(
        np.arange(len(X_train)), test_size=0.1, random_state=42, stratify=y_train
    )

    # Compute group mean from training rows only to avoid leakage
    region_rainfall_mean = X_train.iloc[idx_tr].groupby("Region")["Rainfall_mm"].mean()
    global_mean = X_train.iloc[idx_tr]["Rainfall_mm"].mean()
    X_train["Region_Rainfall_mean"] = X_train["Region"].map(region_rainfall_mean).fillna(global_mean)
    X_val["Region_Rainfall_mean"] = X_val["Region"].map(region_rainfall_mean).fillna(global_mean)

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_cols + categorical_cols

    X_es = X_train.iloc[idx_es][feature_cols]
    y_es = y_train[idx_es]
    X_tr = X_train.iloc[idx_tr][feature_cols]
    y_tr = y_train[idx_tr]

    probas = []
    for seed in [0, 1, 2]:
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
            random_seed=seed,
        )
        model.fit(
            X_tr, y_tr,
            cat_features=categorical_cols,
            eval_set=(X_es, y_es),
            verbose=False,
        )
        probas.append(model.predict_proba(X_val[feature_cols]))
    return np.mean(probas, axis=0)
