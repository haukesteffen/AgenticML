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

HYPOTHESIS = "baseline: cdeotte deterministic logit formula on threshold + growth-stage + mulching indicators"

# Class order matches pd.factorize(sort=True) on ("High", "Low", "Medium"):
# High=0, Low=1, Medium=2.
_INTERCEPTS = np.array([-20.9697, 16.3173, 4.6524])
_COEFS = {
    "soil_lt_25":                   np.array([10.6947, -11.0237,  0.3290]),
    "temp_gt_30":                   np.array([ 5.8763,  -5.8559, -0.0204]),
    "rain_lt_300":                  np.array([10.6958, -10.8500,  0.1542]),
    "wind_gt_10":                   np.array([ 5.7444,  -5.8284,  0.0841]),
    "Crop_Growth_Stage_Flowering":  np.array([ 5.0569,  -5.4155,  0.3586]),
    "Crop_Growth_Stage_Harvest":    np.array([-5.3725,   5.5073, -0.1348]),
    "Crop_Growth_Stage_Sowing":     np.array([-4.8752,   5.2299, -0.3547]),
    "Crop_Growth_Stage_Vegetative": np.array([ 5.1283,  -5.4617,  0.3334]),
    "Mulching_Used_No":             np.array([ 2.8131,  -3.0014,  0.1883]),
    "Mulching_Used_Yes":            np.array([-2.8755,   2.8613,  0.0142]),
}


def _build_indicators(X: pd.DataFrame) -> dict[str, np.ndarray]:
    cols = {
        "soil_lt_25":  (X["Soil_Moisture"]  < 25).to_numpy(dtype=float),
        "temp_gt_30":  (X["Temperature_C"]  > 30).to_numpy(dtype=float),
        "rain_lt_300": (X["Rainfall_mm"]    < 300).to_numpy(dtype=float),
        "wind_gt_10":  (X["Wind_Speed_kmh"] > 10).to_numpy(dtype=float),
    }
    growth = X["Crop_Growth_Stage"].astype(str)
    for stage in ("Flowering", "Harvest", "Sowing", "Vegetative"):
        cols[f"Crop_Growth_Stage_{stage}"] = (growth == stage).to_numpy(dtype=float)
    mulch = X["Mulching_Used"].astype(str)
    for level in ("No", "Yes"):
        cols[f"Mulching_Used_{level}"] = (mulch == level).to_numpy(dtype=float)
    return cols


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Apply cdeotte's deterministic per-class logit formula and softmax.

    The formula is fixed (no fitting); X_train and y_train are unused.
    """
    del X_train, y_train

    indicators = _build_indicators(X_val)
    logits = np.broadcast_to(_INTERCEPTS, (len(X_val), 3)).copy()
    for name, coef in _COEFS.items():
        logits += np.outer(indicators[name], coef)

    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
