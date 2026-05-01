"""Feature recipe layer.

Each recipe is a self-contained module under ``features/`` that exposes:

    def build(train_df, test_df, target, id_column) -> tuple[pd.DataFrame, pd.DataFrame]:
        ...

The returned (X_train, X_test) must be **leakage-safe** — i.e. computed
deterministically from the inputs without using y or test_df labels in a way
that would contaminate train_df. Fold-aware transformations (target encoding,
fold-fitted scalers, OOF features) belong inside ``solution.py``'s
``fit_predict``, not here.

Outputs are cached to ``.cache/features/<name>__<hash>__{train,test}.parquet``
keyed by the recipe file's content hash, so editing a recipe automatically
invalidates its cache.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pandas as pd

CACHE_DIR_NAME = ".cache/features"


def _recipe_module(name: str):
    return importlib.import_module(f"features.{name}")


def _recipe_hash(name: str) -> str:
    mod = _recipe_module(name)
    src = Path(mod.__file__).read_bytes()
    return hashlib.sha256(src).hexdigest()[:12]


def _cache_paths(project_root: Path, name: str, h: str) -> tuple[Path, Path]:
    base = project_root / CACHE_DIR_NAME
    return (
        base / f"{name}__{h}__train.parquet",
        base / f"{name}__{h}__test.parquet",
    )


def load_recipe(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    id_column: str,
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (X_train, X_test) from the named recipe, using a parquet cache."""
    h = _recipe_hash(name)
    train_path, test_path = _cache_paths(project_root, name, h)
    if train_path.exists() and test_path.exists():
        return pd.read_parquet(train_path), pd.read_parquet(test_path)

    mod = _recipe_module(name)
    X_train, X_test = mod.build(train_df, test_df, target=target, id_column=id_column)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(train_path)
    X_test.to_parquet(test_path)
    return X_train, X_test
