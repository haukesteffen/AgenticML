"""
AgenticML solution module.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HYPOTHESIS = "ensembling: logit offset OOF with 2 seeds in inner loop for better OOF quality"

_BASE_PARAMS = dict(tree_method="hist", n_jobs=-1, subsample=0.8, colsample_bytree=0.8, reg_lambda=2, max_bin=2048, n_estimators=250)


def _make_weights(y):
    sw = compute_sample_weight("balanced", y)
    classes, counts = np.unique(y, return_counts=True)
    rarest = classes[np.argmin(counts)]
    sw[y == rarest] *= 4.0
    return sw, rarest


def _apply_offset(proba, rare_idx, offset):
    log_p = np.log(proba + 1e-12)
    log_p[:, rare_idx] += offset
    log_p -= log_p.max(axis=1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=1, keepdims=True)


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    X_train_enc = pd.get_dummies(X_train)
    X_val_enc = pd.get_dummies(X_val)
    X_val_enc = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    sample_weight, rarest = _make_weights(y_train)

    # 3-fold internal OOF to tune rare-class logit offset
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(y_train), 3))
    last_model_inner = None
    for tr, va in skf.split(X_train_enc, y_train):
        sw_tr = _make_weights(y_train[tr])[0]
        for depth in [4, 6]:
            for seed in [0, 1]:
                m = XGBClassifier(**_BASE_PARAMS, max_depth=depth, random_state=seed)
                m.fit(X_train_enc.iloc[tr], y_train[tr], sample_weight=sw_tr)
                oof_proba[va] += m.predict_proba(X_train_enc.iloc[va]) / 4
        last_model_inner = m

    rare_idx = np.where(last_model_inner.classes_ == rarest)[0][0]

    best_offset, best_ba = 0.0, 0.0
    for offset in np.linspace(-1.0, 3.0, 41):
        adj = _apply_offset(oof_proba, rare_idx, offset)
        ba = balanced_accuracy_score(y_train, last_model_inner.classes_[adj.argmax(axis=1)])
        if ba > best_ba:
            best_ba, best_offset = ba, offset

    # Train final models on all training data
    preds = []
    last_model = None
    for depth in [4, 6]:
        for seed in [0, 1]:
            model = XGBClassifier(**_BASE_PARAMS, max_depth=depth, random_state=seed)
            model.fit(X_train_enc, y_train, sample_weight=sample_weight)
            preds.append(model.predict_proba(X_val_enc))
            last_model = model

    mean_proba = np.mean(preds, axis=0)
    rare_idx = np.where(last_model.classes_ == rarest)[0][0]
    return _apply_offset(mean_proba, rare_idx, best_offset)
