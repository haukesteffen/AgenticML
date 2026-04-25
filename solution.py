"""
AgenticML solution module.

Contract
--------
  HYPOTHESIS : str
  fit_predict(X_train, y_train, X_val) -> np.ndarray
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from cuml.neighbors import KNeighborsClassifier

HYPOTHESIS = "cuML KNN k=15, distance weights, numeric-only features"

_NUM_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]


def fit_predict(X_train, y_train, X_val):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train).astype(np.int32)
    n_classes = len(le.classes_)

    X_tr = X_train[_NUM_COLS].fillna(0).astype(np.float32)
    X_vl = X_val[_NUM_COLS].fillna(0).astype(np.float32)

    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr.values).astype(np.float32)
    X_vl_np = scaler.transform(X_vl.values).astype(np.float32)

    knn = KNeighborsClassifier(
        n_neighbors=15, metric="euclidean", weights="distance", output_type="numpy"
    )
    knn.fit(X_tr_np, y_enc)
    probs = knn.predict_proba(X_vl_np)
    return np.array(probs, dtype=np.float64)
