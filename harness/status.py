from __future__ import annotations

import json
import math

import mlflow
from mlflow.entities import ViewType

from harness import git_utils
from harness.config import HarnessConfig


def status(config_path: str = "config.yaml", limit: int = 10) -> None:
    cfg = HarnessConfig.load(config_path)
    branch = git_utils.current_branch(cwd=cfg.project_root)
    experiment_name = f"{cfg.mlflow.experiment_prefix}_{branch}"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(json.dumps({"branch": branch, "experiment": experiment_name, "runs": []}, indent=2))
        return

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
        def _clean(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            return v

        entry = {
            "sha": row.get("tags.sha", ""),
            "status": row.get("tags.status", ""),
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

    output = {
        "branch": branch,
        "experiment": experiment_name,
        "metric": cfg.metric.name,
        "direction": cfg.metric.direction,
        "best_score": best_score,
        "runs": entries,
    }
    print(json.dumps(output, indent=2, default=str))
