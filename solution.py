"""AgenticML solution module."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier

HYPOTHESIS = "TabICL n_estimators=8 default config, cap training context to 504k"

_MAX_CONTEXT = 504_000


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    X_train_enc = pd.get_dummies(X_train)
    X_val_enc = pd.get_dummies(X_val)
    X_val_enc = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    if len(X_train_enc) > _MAX_CONTEXT:
        _, X_train_enc, _, y_train = train_test_split(
            X_train_enc, y_train,
            test_size=_MAX_CONTEXT,
            random_state=42,
            stratify=y_train,
        )

    clf = TabICLClassifier(device="cuda", random_state=0)
    clf.fit(X_train_enc.values.astype(np.float32), y_train)
    return clf.predict_proba(X_val_enc.values.astype(np.float32))
