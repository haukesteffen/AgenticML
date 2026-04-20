from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from harness import mlflow_utils
from harness.config import HarnessConfig


@dataclass(frozen=True)
class SourceSpec:
    alias: str
    branch: str | None = None
    run_id: str | None = None
    selector: str = "best_improved"


@dataclass(frozen=True)
class ResolvedSource:
    alias: str
    run_id: str
    branch: str
    selector: str | None
    hypothesis: str
    mean_score: float | None
    experiment_kind: str


def parse_source_specs(raw_sources: Any) -> list[SourceSpec]:
    if not isinstance(raw_sources, list):
        raise TypeError("SOURCES must be a list of dicts.")

    specs: list[SourceSpec] = []
    seen_aliases: set[str] = set()

    for idx, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise TypeError(f"SOURCES[{idx}] must be a dict, got {type(raw).__name__}.")

        branch = raw.get("branch")
        run_id = raw.get("run_id")
        if bool(branch) == bool(run_id):
            raise ValueError(
                f"SOURCES[{idx}] must specify exactly one of 'branch' or 'run_id'."
            )

        alias_raw = raw.get("alias") or branch or run_id
        alias = _normalize_alias(str(alias_raw))
        if alias in seen_aliases:
            raise ValueError(f"Duplicate source alias {alias!r}.")
        seen_aliases.add(alias)

        selector = str(raw.get("selector", "best_improved"))
        if branch and selector != "best_improved":
            raise ValueError(
                f"SOURCES[{idx}] selector {selector!r} is unsupported. "
                "Use 'best_improved' for branch-based selection."
            )

        specs.append(
            SourceSpec(
                alias=alias,
                branch=str(branch) if branch else None,
                run_id=str(run_id) if run_id else None,
                selector=selector,
            )
        )

    return specs


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


def build_meta_frames(
    cfg: HarnessConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_classes: int,
    raw_sources: Any,
    *,
    classes: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    specs = parse_source_specs(raw_sources)
    if not specs:
        raise ValueError("SOURCES must contain at least one source.")

    resolved = resolve_sources(cfg, specs)
    current_manifest = build_predictions_manifest(
        cfg,
        train_df,
        test_df,
        n_classes,
        _expected_prediction_shape(len(train_df), n_classes, cfg.dataset.problem_type),
        _expected_prediction_shape(len(test_df), n_classes, cfg.dataset.problem_type),
        classes=classes,
    )

    client = mlflow.tracking.MlflowClient()
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    lineage: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for source in resolved:
            source_dir = tmpdir / source.alias
            source_dir.mkdir(parents=True, exist_ok=True)

            oof_preds = _download_and_load_predictions(
                client,
                source,
                artifact="oof_predictions.npy",
                out_dir=source_dir,
                expected_rows=len(train_df),
                n_classes=n_classes,
                problem_type=cfg.dataset.problem_type,
            )
            test_preds = _download_and_load_predictions(
                client,
                source,
                artifact="test_predictions.npy",
                out_dir=source_dir,
                expected_rows=len(test_df),
                n_classes=n_classes,
                problem_type=cfg.dataset.problem_type,
            )

            manifest = _load_required_manifest(client, source, source_dir)
            _validate_manifest(source.alias, manifest, current_manifest)

            train_frames.append(
                _predictions_to_frame(
                    source.alias,
                    oof_preds,
                    index=train_df.index,
                    n_classes=n_classes,
                    problem_type=cfg.dataset.problem_type,
                )
            )
            test_frames.append(
                _predictions_to_frame(
                    source.alias,
                    test_preds,
                    index=test_df.index,
                    n_classes=n_classes,
                    problem_type=cfg.dataset.problem_type,
                )
            )
            lineage.append(
                {
                    "alias": source.alias,
                    "run_id": source.run_id,
                    "branch": source.branch,
                    "selector": source.selector,
                    "hypothesis": source.hypothesis,
                    "mean_score": source.mean_score,
                    "experiment_kind": source.experiment_kind,
                }
            )

    return pd.concat(train_frames, axis=1), pd.concat(test_frames, axis=1), lineage


def _download_and_load_predictions(
    client: mlflow.tracking.MlflowClient,
    source: "ResolvedSource",
    *,
    artifact: str,
    out_dir: Path,
    expected_rows: int,
    n_classes: int,
    problem_type: str,
) -> np.ndarray:
    try:
        path = Path(client.download_artifacts(source.run_id, artifact, str(out_dir)))
    except Exception as e:
        raise RuntimeError(
            f"Source {source.alias!r} (run_id={source.run_id}, branch={source.branch!r}) "
            f"is missing artifact {artifact!r}. Re-run that branch so it produces the "
            f"cached prediction artifacts."
        ) from e

    preds = np.load(str(path))
    return _normalize_source_predictions(
        source.alias,
        preds,
        expected_rows=expected_rows,
        n_classes=n_classes,
        problem_type=problem_type,
    )


def resolve_sources(cfg: HarnessConfig, specs: list[SourceSpec]) -> list[ResolvedSource]:
    client = mlflow.tracking.MlflowClient()
    resolved: list[ResolvedSource] = []

    for spec in specs:
        if spec.run_id:
            run_id = spec.run_id
        else:
            assert spec.branch is not None
            experiment_name = (
                f"{cfg.mlflow.experiment_prefix}_{cfg.mlflow.competition_slug}_{spec.branch}"
            )
            run_id = mlflow_utils.get_best_improved_run_id(
                experiment_name,
                cfg.metric.direction,
                spec.branch,
            )
            if run_id is None:
                raise RuntimeError(
                    f"Could not resolve best improved run for branch {spec.branch!r}."
                )

        run = client.get_run(run_id)
        tags = run.data.tags
        metrics = run.data.metrics
        resolved.append(
            ResolvedSource(
                alias=spec.alias,
                run_id=run_id,
                branch=tags.get("branch", spec.branch or ""),
                selector=None if spec.run_id else spec.selector,
                hypothesis=tags.get("hypothesis", "[missing hypothesis]"),
                mean_score=metrics.get("mean_score"),
                experiment_kind=tags.get("experiment_kind", "model"),
            )
        )

    return resolved


def _normalize_alias(raw_alias: str) -> str:
    alias = raw_alias.strip().replace("/", "_").replace(" ", "_")
    if not alias:
        raise ValueError("Source alias cannot be empty.")
    return alias


def _expected_prediction_shape(
    n_rows: int,
    n_classes: int,
    problem_type: str,
) -> tuple[int, ...]:
    if problem_type == "multiclass_classification":
        return (n_rows, n_classes)
    return (n_rows,)


def _normalize_source_predictions(
    alias: str,
    preds: np.ndarray,
    *,
    expected_rows: int,
    n_classes: int,
    problem_type: str,
) -> np.ndarray:
    preds = np.asarray(preds)

    if problem_type == "multiclass_classification":
        if preds.ndim != 2 or preds.shape != (expected_rows, n_classes):
            raise ValueError(
                f"Source {alias!r} has invalid multiclass OOF shape {preds.shape}; "
                f"expected ({expected_rows}, {n_classes})."
            )
    else:
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds[:, 0]
        if preds.ndim != 1 or preds.shape[0] != expected_rows:
            raise ValueError(
                f"Source {alias!r} has invalid OOF shape {preds.shape}; "
                f"expected ({expected_rows},)."
            )

    if np.any(np.isnan(preds)):
        raise ValueError(f"Source {alias!r} OOF predictions contain NaN.")
    if np.any(np.isinf(preds)):
        raise ValueError(f"Source {alias!r} OOF predictions contain inf.")

    return preds


def _predictions_to_frame(
    alias: str,
    preds: np.ndarray,
    *,
    index: pd.Index,
    n_classes: int,
    problem_type: str,
) -> pd.DataFrame:
    if problem_type == "multiclass_classification":
        cols = [f"{alias}__class_{i}" for i in range(n_classes)]
        return pd.DataFrame(preds, columns=cols, index=index)
    return pd.DataFrame({f"{alias}__pred": preds}, index=index)


def _load_required_manifest(
    client: mlflow.tracking.MlflowClient,
    source: "ResolvedSource",
    out_dir: Path,
) -> dict[str, object]:
    try:
        manifest_path = Path(
            client.download_artifacts(source.run_id, "predictions_manifest.json", str(out_dir))
        )
    except Exception as e:
        raise RuntimeError(
            f"Source {source.alias!r} (run_id={source.run_id}, branch={source.branch!r}) "
            f"is missing predictions_manifest.json. Re-run that branch so it produces "
            f"the manifest alongside its predictions."
        ) from e

    return json.loads(manifest_path.read_text())


def _validate_manifest(
    alias: str,
    source_manifest: dict[str, object],
    current_manifest: dict[str, object],
) -> None:
    checks = (
        "problem_type",
        "train_path",
        "test_path",
        "target",
        "id_column",
        "n_rows",
        "n_test_rows",
        "n_classes",
        "dataset_signature",
        "test_dataset_signature",
        "class_values",
    )
    for key in checks:
        if source_manifest.get(key) != current_manifest.get(key):
            raise ValueError(
                f"Source {alias!r} manifest mismatch for {key!r}: "
                f"{source_manifest.get(key)!r} != {current_manifest.get(key)!r}"
            )

    if source_manifest.get("cv") != current_manifest.get("cv"):
        raise ValueError(
            f"Source {alias!r} manifest mismatch for 'cv': "
            f"{source_manifest.get('cv')!r} != {current_manifest.get('cv')!r}"
        )
