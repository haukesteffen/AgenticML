"""Resolve a list of lane names to their promoted artifacts.

Phase e's ``ensemble.py`` declares ``SOURCES`` as a list of lane strings (e.g.
``"v1_raw__LGBMClassifier"``). For each lane we pick the latest ``promoted``
run in the ``{prefix}_{slug}_promoted`` MLflow experiment and load its
``oof_outer.npy``, ``oof_inner.npy``, and ``test_pred.npy`` artifacts.

Downloads are cached under ``.cache/promoted/<run_id>__*.npy`` keyed by run id
so subsequent ensembling iterations are fast.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from mlflow.entities import ViewType

from harness.config import HarnessConfig

CACHE_DIR_NAME = ".cache/promoted"


@dataclass(frozen=True)
class PromotedSource:
    lane: str
    run_id: str
    recipe: str
    family: str
    source_branch: str
    source_commit: str
    cv_score_outer: float | None
    outer_oof: np.ndarray  # (N,) or (N, C)
    inner_oof: np.ndarray  # (N, K) or (N, K, C); NaN where row was outer-holdout for that fold
    test_pred: np.ndarray  # (N_test,) or (N_test, C)


def _promoted_experiment_name(cfg: HarnessConfig) -> str:
    return f"{cfg.mlflow.experiment_prefix}_{cfg.mlflow.competition_slug}_promoted"


def _latest_run_for_lane(cfg: HarnessConfig, lane: str) -> mlflow.entities.Run:
    experiment = mlflow.get_experiment_by_name(_promoted_experiment_name(cfg))
    if experiment is None:
        raise RuntimeError(
            f"Promoted experiment {_promoted_experiment_name(cfg)!r} does not exist. "
            f"Promote at least one lane first via `python -m harness promote`."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.lane = '{lane}' AND tags.status = 'promoted'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"No promoted runs found for lane {lane!r}. "
            f"Run `python -m harness promote` on that lane's branch first."
        )
    run_id = runs.iloc[0]["run_id"]
    return mlflow.tracking.MlflowClient().get_run(run_id)


def _cached_load(
    project_root: Path,
    client: mlflow.tracking.MlflowClient,
    run_id: str,
    artifact: str,
    cache_suffix: str,
) -> np.ndarray:
    cache_dir = project_root / CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{run_id}__{cache_suffix}.npy"
    if cache_path.exists():
        return np.load(str(cache_path))
    download_dir = cache_dir / f"_download_{run_id}"
    download_dir.mkdir(exist_ok=True)
    src = Path(client.download_artifacts(run_id, artifact, str(download_dir)))
    arr = np.load(str(src))
    np.save(str(cache_path), arr)
    return arr


def resolve_sources(cfg: HarnessConfig, lanes: list[str]) -> list[PromotedSource]:
    if not lanes:
        raise ValueError("SOURCES must contain at least one lane name.")
    if len(set(lanes)) != len(lanes):
        raise ValueError("SOURCES contains duplicate lane names.")

    client = mlflow.tracking.MlflowClient()
    out: list[PromotedSource] = []
    for lane in lanes:
        run = _latest_run_for_lane(cfg, lane)
        run_id = run.info.run_id
        tags = run.data.tags
        out.append(
            PromotedSource(
                lane=lane,
                run_id=run_id,
                recipe=tags.get("recipe", ""),
                family=tags.get("family", ""),
                source_branch=tags.get("source_branch", ""),
                source_commit=tags.get("source_commit", ""),
                cv_score_outer=run.data.metrics.get("cv_score_outer"),
                outer_oof=_cached_load(cfg.project_root, client, run_id, "oof_outer.npy", "outer"),
                inner_oof=_cached_load(cfg.project_root, client, run_id, "oof_inner.npy", "inner"),
                test_pred=_cached_load(cfg.project_root, client, run_id, "test_pred.npy", "test"),
            )
        )
    return out
