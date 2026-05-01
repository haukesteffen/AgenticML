"""Shared helpers used by both base-model and ensemble workers."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from harness.config import HarnessConfig


def build_predictions_manifest(
    cfg: HarnessConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_classes: int,
    oof_shape: tuple[int, ...],
    test_shape: tuple[int, ...],
    *,
    classes: Any = None,
) -> dict[str, object]:
    train_columns = [cfg.dataset.target]
    if cfg.dataset.id_column in train_df.columns:
        train_columns.insert(0, cfg.dataset.id_column)
    train_hashed = pd.util.hash_pandas_object(train_df[train_columns], index=False).values
    dataset_signature = hashlib.sha256(train_hashed.tobytes()).hexdigest()

    test_hashed = pd.util.hash_pandas_object(test_df[[cfg.dataset.id_column]], index=False).values
    test_dataset_signature = hashlib.sha256(test_hashed.tobytes()).hexdigest()

    return {
        "schema_version": 2,
        "problem_type": cfg.dataset.problem_type,
        "train_path": cfg.dataset.train_path,
        "test_path": cfg.dataset.test_path,
        "target": cfg.dataset.target,
        "id_column": cfg.dataset.id_column,
        "n_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_classes": n_classes,
        "oof_shape": [int(dim) for dim in oof_shape],
        "test_shape": [int(dim) for dim in test_shape],
        "cv": {
            "n_splits": cfg.cv.n_splits,
            "shuffle": cfg.cv.shuffle,
            "seed": cfg.cv.seed,
        },
        "dataset_signature": dataset_signature,
        "test_dataset_signature": test_dataset_signature,
        "class_values": None if classes is None else [str(value) for value in classes],
    }


def log_json_artifact(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
    filename: str,
    payload: dict[str, object] | list[object],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        client.log_artifact(run_id, str(path))
