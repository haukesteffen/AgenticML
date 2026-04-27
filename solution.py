"""
AgenticML solution module.

The harness calls ``fit_predict`` once per fold, passing only that fold's
training slice and validation slice. All preprocessing is fit on ``X_train``
inside ``fit_predict`` to avoid cross-fold leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import TargetEncoder

HYPOTHESIS = "ensembling: two seed XGBoost blend"

SEED = 2026
DIGIT_POSITIONS = range(-4, 4)
BLEND_SEEDS = [0, 1]
BEST_PARAMS = {
    "learning_rate": 0.1894375940266229,
    "max_depth": 4,
    "min_child_weight": 9.026983103702738,
    "subsample": 0.7086120837262754,
    "colsample_bytree": 0.9056257267916479,
    "gamma": 0.001331367288709093,
    "reg_alpha": 0.0008988873979690247,
    "reg_lambda": 5.765133195499853,
}


def _add_digit_features(
    X: pd.DataFrame,
    numeric_cols: list[str],
    max_values: pd.Series,
) -> pd.DataFrame:
    X = X.copy()
    digit_parts = {}

    for col in numeric_cols:
        for k in DIGIT_POSITIONS:
            digit_parts[f"{col}_digit{k}"] = (X[col] // (10 ** k) % 10).astype("int8")

        if max_values[col] < 10:
            X[col] = X[col].round(3)
        elif max_values[col] < 100:
            X[col] = X[col].round(2)
        else:
            X[col] = X[col].round(1)

    if digit_parts:
        X = pd.concat([X, pd.DataFrame(digit_parts, index=X.index)], axis=1)
    return X


def _frequency_encode(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    category_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_val = X_val.copy()

    for col in category_cols:
        freq = X_train[col].value_counts()
        mapping = {
            val: idx
            for idx, (val, _count) in enumerate(freq[freq >= 5].items())
        }
        mapping_default = len(mapping)
        X_train[col] = X_train[col].map(lambda x: mapping.get(x, mapping_default))
        X_val[col] = X_val[col].map(lambda x: mapping.get(x, mapping_default))

    return X_train, X_val


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train on one harness fold and return validation probabilities."""
    y_train = np.asarray(y_train)
    n_classes = int(np.max(y_train)) + 1

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    max_values = X_train[numeric_cols].max()
    X_train_fe = _add_digit_features(X_train, numeric_cols, max_values)
    X_val_fe = _add_digit_features(X_val, numeric_cols, max_values)

    drop_cols = [col for col in X_val_fe.columns if X_val_fe[col].nunique() == 1]
    if drop_cols:
        X_train_fe = X_train_fe.drop(columns=drop_cols)
        X_val_fe = X_val_fe.drop(columns=drop_cols)

    categorical_cols = [col for col in categorical_cols if col not in drop_cols]
    numeric_cols = [col for col in numeric_cols if col not in drop_cols]
    category_cols = categorical_cols + [col for col in X_val_fe.columns if "digit" in col]
    target_encoder_cols = category_cols + numeric_cols

    X_train_fe, X_val_fe = _frequency_encode(X_train_fe, X_val_fe, category_cols)

    te = TargetEncoder(target_type="multiclass", smooth="auto", cv=5, random_state=42)
    X_train_te = te.fit_transform(X_train_fe[target_encoder_cols], y_train)
    X_val_te = te.transform(X_val_fe[target_encoder_cols])

    te_cols = [f"te_{i}" for i in range(X_train_te.shape[1])]
    X_train_te = pd.DataFrame(X_train_te, index=X_train_fe.index, columns=te_cols)
    X_val_te = pd.DataFrame(X_val_te, index=X_val_fe.index, columns=te_cols)

    X_train_model = pd.concat([X_train_fe, X_train_te], axis=1).drop(columns=categorical_cols)
    X_val_model = pd.concat([X_val_fe, X_val_te], axis=1).drop(columns=categorical_cols)

    class_values, class_counts = np.unique(y_train, return_counts=True)
    avg_count = len(y_train) / len(class_values)
    weights = {cls: avg_count / count for cls, count in zip(class_values, class_counts)}
    sample_weights = np.array([weights[cls] for cls in y_train])

    proba_sum = np.zeros((len(X_val_model), n_classes), dtype=float)
    for seed in BLEND_SEEDS:
        model = xgb.XGBClassifier(
            **BEST_PARAMS,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train_model, y_train, sample_weight=sample_weights, verbose=False)

        proba = model.predict_proba(X_val_model)
        if proba.shape[1] != n_classes:
            aligned = np.zeros((len(X_val_model), n_classes), dtype=float)
            for i, cls in enumerate(model.classes_):
                aligned[:, int(cls)] = proba[:, i]
            proba = aligned
        proba_sum += proba

    return proba_sum / len(BLEND_SEEDS)
