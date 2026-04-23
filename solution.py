"""
AgenticML solution module.
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HYPOTHESIS = "hyperparams: balanced weights + 4.0x multiplier on rarest class (High)"

_BASE_PARAMS = dict(tree_method="hist", n_jobs=-1, subsample=0.8, colsample_bytree=0.8, reg_lambda=2, max_bin=2048, n_estimators=150)


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    sample_weight = compute_sample_weight("balanced", y_train)
    classes, counts = np.unique(y_train, return_counts=True)
    rarest = classes[np.argmin(counts)]
    sample_weight[y_train == rarest] *= 4.0

    preds = []
    for depth in [4, 6]:
        for seed in [0, 1]:
            model = XGBClassifier(**_BASE_PARAMS, max_depth=depth, random_state=seed)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            preds.append(model.predict_proba(X_val))
    return np.mean(preds, axis=0)
