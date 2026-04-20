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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

HYPOTHESIS = "add lightgbm source to enriched logistic stacker ensemble"

SOURCES = [
    {"alias": "catboost", "branch": "exp/catboost", "selector": "best_improved"},
    {"alias": "lightgbm", "branch": "exp/lightgbm", "selector": "best_improved"},
    {"alias": "lightgbm2", "branch": "exp/lightgbm2", "selector": "best_improved"},
    {"alias": "mlp2", "branch": "exp/mlp2", "selector": "best_improved"},
    {"alias": "realmlp", "branch": "exp/realmlp", "selector": "best_improved"},
    {"alias": "xgb", "branch": "exp/xgb", "selector": "best_improved"},
]


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Fit logistic stacking, then class-prior calibrate probabilities."""
    if not SOURCES:
        raise ValueError("SOURCES is empty. Add at least one source run before running the harness.")

    X_train_np = X_train.to_numpy(dtype=float)
    X_val_np = X_val.to_numpy(dtype=float)
    eps = 1e-6
    X_train_log_odds = np.log(np.clip(X_train_np, eps, 1.0 - eps) / np.clip(1.0 - X_train_np, eps, 1.0))
    X_val_log_odds = np.log(np.clip(X_val_np, eps, 1.0 - eps) / np.clip(1.0 - X_val_np, eps, 1.0))

    source_prefixes = sorted(
        {
            column_name.rsplit("__class_", maxsplit=1)[0]
            for column_name in X_train.columns
            if "__class_" in column_name
        }
    )
    extra_train = []
    extra_val = []
    train_votes = []
    val_votes = []
    train_source_blocks = []
    val_source_blocks = []
    for prefix in source_prefixes:
        source_cols = sorted(
            [idx for idx, col in enumerate(X_train.columns) if col.startswith(f"{prefix}__class_")],
            key=lambda idx: int(X_train.columns[idx].rsplit("__class_", maxsplit=1)[1]),
        )
        if len(source_cols) < 2:
            continue
        train_source_proba = np.clip(X_train_np[:, source_cols], eps, 1.0)
        val_source_proba = np.clip(X_val_np[:, source_cols], eps, 1.0)

        train_entropy = -np.sum(train_source_proba * np.log(train_source_proba), axis=1, keepdims=True)
        val_entropy = -np.sum(val_source_proba * np.log(val_source_proba), axis=1, keepdims=True)

        train_top2 = np.sort(np.partition(train_source_proba, -2, axis=1)[:, -2:], axis=1)
        val_top2 = np.sort(np.partition(val_source_proba, -2, axis=1)[:, -2:], axis=1)
        train_margin = (train_top2[:, 1] - train_top2[:, 0]).reshape(-1, 1)
        val_margin = (val_top2[:, 1] - val_top2[:, 0]).reshape(-1, 1)

        extra_train.extend([train_entropy, train_margin])
        extra_val.extend([val_entropy, val_margin])
        train_votes.append(train_source_proba.argmax(axis=1))
        val_votes.append(val_source_proba.argmax(axis=1))
        train_source_blocks.append(train_source_proba)
        val_source_blocks.append(val_source_proba)

    if train_votes:
        train_vote_labels = np.column_stack(train_votes)
        val_vote_labels = np.column_stack(val_votes)
        n_classes = X_train_np.shape[1] // max(len(source_prefixes), 1)
        train_vote_frac = np.column_stack(
            [(train_vote_labels == cls).mean(axis=1) for cls in range(n_classes)]
        )
        val_vote_frac = np.column_stack(
            [(val_vote_labels == cls).mean(axis=1) for cls in range(n_classes)]
        )
        extra_train.append(train_vote_frac)
        extra_val.append(val_vote_frac)

    if train_source_blocks:
        train_source_tensor = np.stack(train_source_blocks, axis=1)
        val_source_tensor = np.stack(val_source_blocks, axis=1)
        train_consensus_std = np.std(train_source_tensor, axis=1)
        val_consensus_std = np.std(val_source_tensor, axis=1)
        train_consensus_max = np.max(train_source_tensor, axis=1)
        val_consensus_max = np.max(val_source_tensor, axis=1)
        extra_train.extend([train_consensus_std, train_consensus_max])
        extra_val.extend([val_consensus_std, val_consensus_max])

    # Strong-pair disagreement signal: per-class probability deltas.
    pair_left = "catboost"
    pair_right = "xgb"
    left_cols = [idx for idx, col in enumerate(X_train.columns) if col.startswith(f"{pair_left}__class_")]
    right_cols = [idx for idx, col in enumerate(X_train.columns) if col.startswith(f"{pair_right}__class_")]
    if left_cols and len(left_cols) == len(right_cols):
        left_cols = sorted(left_cols, key=lambda idx: int(X_train.columns[idx].rsplit("__class_", maxsplit=1)[1]))
        right_cols = sorted(right_cols, key=lambda idx: int(X_train.columns[idx].rsplit("__class_", maxsplit=1)[1]))
        extra_train.append(X_train_np[:, left_cols] - X_train_np[:, right_cols])
        extra_val.append(X_val_np[:, left_cols] - X_val_np[:, right_cols])

    if extra_train:
        X_train_np = np.hstack([X_train_log_odds, *extra_train])
        X_val_np = np.hstack([X_val_log_odds, *extra_val])
    else:
        X_train_np = X_train_log_odds
        X_val_np = X_val_log_odds
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_np,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    candidate_cs = [0.25, 0.5, 1.0, 2.0, 4.0]
    best_c = 1.0
    best_c_score = -np.inf
    best_cal_proba = None
    for c_val in candidate_cs:
        candidate_model = LogisticRegression(
            C=c_val,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        candidate_model.fit(X_fit, y_fit)
        candidate_proba = candidate_model.predict_proba(X_cal)
        candidate_pred = candidate_proba.argmax(axis=1)
        candidate_score = balanced_accuracy_score(y_cal, candidate_pred)
        if candidate_score > best_c_score:
            best_c_score = candidate_score
            best_c = float(c_val)
            best_cal_proba = candidate_proba

    cal_proba = best_cal_proba

    model = LogisticRegression(
        C=best_c,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_train_np, y_train)
    val_proba = model.predict_proba(X_val_np)
    if val_proba.ndim == 2 and val_proba.shape[1] == 1:
        return val_proba[:, 0]

    class_counts = np.bincount(y_train, minlength=val_proba.shape[1]).astype(float)
    class_priors = np.clip(class_counts / class_counts.sum(), 1e-6, 1.0)

    best_gamma = 0.0
    best_score = -np.inf
    for gamma in np.linspace(0.0, 1.5, 16):
        adjusted = cal_proba / np.power(class_priors, gamma)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        pred_labels = adjusted.argmax(axis=1)
        score = balanced_accuracy_score(y_cal, pred_labels)
        if score > best_score:
            best_score = score
            best_gamma = float(gamma)

    val_adjusted = val_proba / np.power(class_priors, best_gamma)
    val_adjusted /= val_adjusted.sum(axis=1, keepdims=True)
    return val_adjusted
