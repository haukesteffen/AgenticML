"""
AgenticML solution module.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HYPOTHESIS = "baseline: vanilla XGBClassifier"


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)
