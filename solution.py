"""
AgenticML solution module.
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HYPOTHESIS = "ensembling: seed bagging over 3 XGBs (different seeds) for variance reduction"


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    sample_weight = compute_sample_weight("balanced", y_train)

    preds = []
    for seed in [0, 1, 2]:
        model = XGBClassifier(tree_method="hist", n_jobs=-1, subsample=0.8, colsample_bytree=0.8, random_state=seed)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        preds.append(model.predict_proba(X_val))
    return np.mean(preds, axis=0)
