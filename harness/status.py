from __future__ import annotations

import json
import math

import mlflow
from mlflow.entities import ViewType

from harness import git_utils
from harness.config import HarnessConfig


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def _resolve_experiment_name(cfg: HarnessConfig, branch: str, experiment: str | None) -> str:
    prefix = f"{cfg.mlflow.experiment_prefix}_{cfg.mlflow.competition_slug}"
    if experiment is None:
        return f"{prefix}_{branch}"
    if experiment == "promoted":
        return f"{prefix}_promoted"
    return experiment


def _branch_view(cfg: HarnessConfig, branch: str, experiment_name: str, limit: int) -> dict:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {"branch": branch, "experiment": experiment_name, "runs": []}

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.branch = '{branch}'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["start_time DESC"],
        max_results=limit,
    )

    if "tags.mlflow.parentRunId" in runs_df.columns:
        parent_runs = runs_df[runs_df["tags.mlflow.parentRunId"].isna()].copy()
    else:
        parent_runs = runs_df.copy()

    entries = []
    for _, row in parent_runs.iterrows():
        entry = {
            "sha": row.get("tags.sha", ""),
            "status": row.get("tags.status", ""),
            "experiment_kind": row.get("tags.experiment_kind", "model"),
            "recipe": row.get("tags.recipe", ""),
            "hypothesis": row.get("tags.hypothesis", ""),
            "mean_score": _clean(row.get("metrics.mean_score")),
            "std_score": _clean(row.get("metrics.std_score")),
            "duration_s": None,
        }
        if row.get("start_time") and row.get("end_time"):
            try:
                duration = (row["end_time"] - row["start_time"]).total_seconds()
                entry["duration_s"] = round(duration, 1)
            except Exception:
                pass
        entries.append(entry)

    best_improved = [
        e for e in entries
        if e["status"] == "improved" and e["mean_score"] is not None
    ]
    best_score = None
    if best_improved:
        if cfg.metric.direction == "maximize":
            best_score = max(e["mean_score"] for e in best_improved)
        else:
            best_score = min(e["mean_score"] for e in best_improved)

    return {
        "branch": branch,
        "experiment": experiment_name,
        "metric": cfg.metric.name,
        "direction": cfg.metric.direction,
        "best_score": best_score,
        "runs": entries,
    }


def _promoted_view(cfg: HarnessConfig, experiment_name: str, lane: str | None, limit: int) -> dict:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {"experiment": experiment_name, "lanes": []}

    filter_parts = ["tags.status = 'promoted'"]
    if lane is not None:
        filter_parts.append(f"tags.lane = '{lane}'")
    filter_string = " AND ".join(filter_parts)

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["start_time DESC"],
        max_results=limit if lane else max(limit, 200),
    )

    entries = []
    for _, row in runs_df.iterrows():
        entries.append({
            "lane": row.get("tags.lane", ""),
            "recipe": row.get("tags.recipe", ""),
            "family": row.get("tags.family", ""),
            "cv_score_outer": _clean(row.get("metrics.cv_score_outer")),
            "source_branch": row.get("tags.source_branch", ""),
            "source_commit": (row.get("tags.source_commit") or "")[:8],
            "promoted_at": row.get("tags.promoted_at", ""),
            "run_id": row.get("run_id", ""),
            "hypothesis": row.get("tags.hypothesis", ""),
        })

    if lane is not None:
        return {
            "experiment": experiment_name,
            "lane": lane,
            "history": entries,
        }

    # Group by lane, keep best per lane
    direction = cfg.metric.direction
    by_lane: dict[str, dict] = {}
    for e in entries:
        if e["cv_score_outer"] is None:
            continue
        cur = by_lane.get(e["lane"])
        if cur is None:
            by_lane[e["lane"]] = e
            continue
        if direction == "maximize" and e["cv_score_outer"] > cur["cv_score_outer"]:
            by_lane[e["lane"]] = e
        elif direction == "minimize" and e["cv_score_outer"] < cur["cv_score_outer"]:
            by_lane[e["lane"]] = e

    return {
        "experiment": experiment_name,
        "metric": cfg.metric.name,
        "direction": cfg.metric.direction,
        "lanes": sorted(by_lane.values(), key=lambda x: x["lane"]),
    }


def status(
    config_path: str = "config.yaml",
    limit: int = 10,
    *,
    experiment: str | None = None,
    lane: str | None = None,
) -> None:
    cfg = HarnessConfig.load(config_path)
    branch = git_utils.current_branch(cwd=cfg.project_root)
    experiment_name = _resolve_experiment_name(cfg, branch, experiment)

    if experiment == "promoted" or lane is not None:
        if experiment is None:
            experiment_name = _resolve_experiment_name(cfg, branch, "promoted")
        out = _promoted_view(cfg, experiment_name, lane, limit)
    else:
        out = _branch_view(cfg, branch, experiment_name, limit)

    print(json.dumps(out, indent=2, default=str))
