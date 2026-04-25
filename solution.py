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
from sklearn.preprocessing import SplineTransformer
from sklearn.decomposition import PCA
from cuml.neighbors import KNeighborsClassifier

HYPOTHESIS = "cuML KNN multi-k (50,100,200) + spline(n_knots=10) + PCA(8,16,32,64) + 3-seed oversample ensemble"

_NUM_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
_CAT_COLS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]


def _target_encode(X_tr, y_enc, X_vl, n_classes, smoothing=30):
    global_mean = np.bincount(y_enc, minlength=n_classes) / len(y_enc)
    tr_enc = np.zeros((len(X_tr), len(_CAT_COLS) * n_classes), dtype=np.float32)
    vl_enc = np.zeros((len(X_vl), len(_CAT_COLS) * n_classes), dtype=np.float32)
    for ci, col in enumerate(_CAT_COLS):
        col_offset = ci * n_classes
        cats_tr = X_tr[col].fillna("__NaN__").astype(str).values
        cats_vl = X_vl[col].fillna("__NaN__").astype(str).values
        cat_map = {}
        for cat in np.unique(cats_tr):
            mask = cats_tr == cat
            n = mask.sum()
            local = np.bincount(y_enc[mask], minlength=n_classes) / n
            w = n / (n + smoothing)
            cat_map[cat] = w * local + (1 - w) * global_mean
        for row_i, cat in enumerate(cats_tr):
            tr_enc[row_i, col_offset:col_offset + n_classes] = cat_map[cat]
        for row_i, cat in enumerate(cats_vl):
            vl_enc[row_i, col_offset:col_offset + n_classes] = cat_map.get(cat, global_mean)
    return tr_enc, vl_enc


def _run_knn(X_tr, y, X_vl, ks=(50, 100, 200)):
    all_probs = []
    for k in ks:
        knn = KNeighborsClassifier(
            n_neighbors=k, metric="euclidean", weights="distance", output_type="numpy"
        )
        knn.fit(X_tr, y)
        all_probs.append(knn.predict_proba(X_vl))
    return np.mean(all_probs, axis=0)


def fit_predict(X_train, y_train, X_val):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train).astype(np.int32)
    n_classes = len(le.classes_)

    X_tr_num = X_train[_NUM_COLS].fillna(0).values.astype(np.float32)
    X_vl_num = X_val[_NUM_COLS].fillna(0).values.astype(np.float32)

    spline = SplineTransformer(n_knots=10, degree=3, include_bias=False)
    X_tr_spline = spline.fit_transform(X_tr_num).astype(np.float32)
    X_vl_spline = spline.transform(X_vl_num).astype(np.float32)

    tr_cat, vl_cat = _target_encode(X_train, y_enc, X_val, n_classes)
    X_tr_raw = np.hstack([X_tr_spline, tr_cat])
    X_vl_raw = np.hstack([X_vl_spline, vl_cat])

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_vl_scaled = scaler.transform(X_vl_raw).astype(np.float32)

    counts = np.bincount(y_enc, minlength=n_classes)
    max_count = counts.max()

    all_probs = []
    for seed in [42, 123, 777]:
        rng = np.random.default_rng(seed)
        for n_components in [8, 16, 32, 64]:
            pca = PCA(n_components=n_components, random_state=seed)
            X_tr_pca = pca.fit_transform(X_tr_scaled).astype(np.float32)
            X_vl_pca = pca.transform(X_vl_scaled).astype(np.float32)

            extra_X, extra_y = [], []
            for cls in range(n_classes):
                if counts[cls] < max_count:
                    idx = np.where(y_enc == cls)[0]
                    n_extra = max_count - counts[cls]
                    chosen = rng.choice(idx, size=n_extra, replace=True)
                    noise = rng.normal(0, 0.01, size=(n_extra, n_components)).astype(np.float32)
                    extra_X.append(X_tr_pca[chosen] + noise)
                    extra_y.append(np.full(n_extra, cls, dtype=np.int32))
            if extra_X:
                X_tr_bal = np.vstack([X_tr_pca] + extra_X)
                y_bal = np.concatenate([y_enc] + extra_y)
            else:
                X_tr_bal, y_bal = X_tr_pca, y_enc

            all_probs.append(_run_knn(X_tr_bal, y_bal, X_vl_pca))

    probs = np.mean(all_probs, axis=0)
    return np.array(probs, dtype=np.float64)
