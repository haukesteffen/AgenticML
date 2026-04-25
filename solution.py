"""
AgenticML solution module.

Contract
--------
  HYPOTHESIS : str
  fit_predict(X_train, y_train, X_val) -> np.ndarray
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from cuml.neighbors import KNeighborsClassifier

HYPOTHESIS = "cuML GPU KNN k=50, euclidean, StandardScaler + get_dummies"


def fit_predict(X_train, y_train, X_val):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train).astype(np.int32)
    n_classes = len(le.classes_)

    X_tr = pd.get_dummies(X_train).fillna(0).astype(np.float32)
    X_vl = pd.get_dummies(X_val).fillna(0).astype(np.float32)
    X_vl = X_vl.reindex(columns=X_tr.columns, fill_value=0)

    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr.values).astype(np.float32)
    X_vl_np = scaler.transform(X_vl.values).astype(np.float32)

    knn = KNeighborsClassifier(n_neighbors=50, metric="euclidean", output_type="numpy")
    knn.fit(X_tr_np, y_enc)

    probs = knn.predict_proba(X_vl_np)
    if probs.shape[1] != n_classes:
        # Reorder columns to match label encoder order
        out = np.zeros((len(X_vl_np), n_classes), dtype=np.float32)
        for i, c in enumerate(knn.classes_):
            out[:, le.transform([c])[0]] = probs[:, i]
        probs = out

    return probs.astype(np.float64)
