from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def _to_hard_labels(preds: np.ndarray) -> np.ndarray:
    if preds.ndim == 2:
        return preds.argmax(axis=1)
    return (preds >= 0.5).astype(int)


def _roc_auc(y_true: np.ndarray, preds: np.ndarray, problem_type: str) -> float:
    if problem_type == "multiclass_classification":
        return roc_auc_score(y_true, preds, multi_class="ovr")
    return roc_auc_score(y_true, preds)


def _accuracy(y_true: np.ndarray, preds: np.ndarray, problem_type: str) -> float:
    return accuracy_score(y_true, _to_hard_labels(preds))


def _balanced_accuracy(y_true: np.ndarray, preds: np.ndarray, problem_type: str) -> float:
    return balanced_accuracy_score(y_true, _to_hard_labels(preds))


def _logloss(y_true: np.ndarray, preds: np.ndarray, problem_type: str) -> float:
    return log_loss(y_true, preds)


def _rmse(y_true: np.ndarray, preds: np.ndarray, problem_type: str) -> float:
    return mean_squared_error(y_true, preds, squared=False)


def _mae(y_true: np.ndarray, preds: np.ndarray, problem_type: str) -> float:
    return mean_absolute_error(y_true, preds)


MetricFn = Callable[[np.ndarray, np.ndarray, str], float]

METRICS: dict[str, tuple[MetricFn, str]] = {
    "roc_auc": (_roc_auc, "maximize"),
    "accuracy": (_accuracy, "maximize"),
    "balanced_accuracy": (_balanced_accuracy, "maximize"),
    "logloss": (_logloss, "minimize"),
    "rmse": (_rmse, "minimize"),
    "mae": (_mae, "minimize"),
}


def get_metric(name: str) -> tuple[MetricFn, str]:
    if name not in METRICS:
        raise ValueError(f"Unknown metric {name!r}. Choose from: {sorted(METRICS)}")
    return METRICS[name]
