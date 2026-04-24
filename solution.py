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
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

HYPOTHESIS = "in-sample log-prob offset tuning per class to directly maximize balanced_accuracy"


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train a model on (X_train, y_train) and return predictions on X_val."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Oversample the rare "High" class (index 0 after pd.factorize sort=True: High < Low < Medium)
    high_mask = y_train == 0
    X_aug = pd.concat([X_train] + [X_train[high_mask]] * 5, ignore_index=True)
    y_aug = np.concatenate([y_train] + [y_train[high_mask]] * 5)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", TargetEncoder(random_state=42), categorical_cols),
    ])

    fitted_pipes, val_probas = [], []
    for seed in [42, 7, 123, 17, 99]:
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", MLPClassifier(max_iter=200, alpha=1e-6, random_state=seed)),
        ])
        pipe.fit(X_aug, y_aug)
        fitted_pipes.append(pipe)
        val_probas.append(pipe.predict_proba(X_val))

    # Find per-class log-prob offsets that maximize balanced_accuracy on training fold
    # Using original X_train (not augmented) so in-sample eval is on natural distribution
    train_probas = np.mean([p.predict_proba(X_train) for p in fitted_pipes], axis=0)

    def neg_bal_acc(offsets):
        # offsets: 2 free params; fix class 2 (Medium) to 0 for identifiability
        w = np.exp([offsets[0], offsets[1], 0.0])
        return -balanced_accuracy_score(y_train, np.argmax(train_probas * w, axis=1))

    result = minimize(neg_bal_acc, [0.0, 0.0], method="Nelder-Mead",
                      options={"maxiter": 300, "xatol": 1e-5})
    opt_w = np.exp([result.x[0], result.x[1], 0.0])

    raw_val = np.mean(val_probas, axis=0)
    adj_val = raw_val * opt_w
    adj_val /= adj_val.sum(axis=1, keepdims=True)
    return adj_val
