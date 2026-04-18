from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow.entities import ViewType


def setup_autolog() -> None:
    mlflow.sklearn.autolog(
        log_models=False,
        log_datasets=False,
        log_input_examples=False,
        log_model_signatures=False,
        log_post_training_metrics=False,
        silent=True,
    )


def ensure_experiment(experiment_name: str) -> str:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
    exp = mlflow.set_experiment(experiment_name)
    return exp.experiment_id


def start_parent_run(
    experiment_name: str,
    tags: dict[str, str],
) -> mlflow.ActiveRun:
    ensure_experiment(experiment_name)
    run = mlflow.start_run(tags=tags)
    return run


def end_parent_run(status: str) -> None:
    mlflow.set_tag("status", status)
    mlflow.end_run()


def log_traceback_artifact(traceback_text: str, filename: str = "traceback.txt") -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        path.write_text(traceback_text)
        mlflow.log_artifact(str(path))


def get_best_score(
    experiment_name: str,
    direction: str,
    branch: str,
) -> float | None:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.status = 'improved' AND tags.branch = '{branch}'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    if runs.empty or "metrics.mean_score" not in runs.columns:
        return None

    scores = runs["metrics.mean_score"].dropna()
    if scores.empty:
        return None

    return scores.max() if direction == "maximize" else scores.min()


def get_parent_mean_score(run_id: str) -> float | None:
    run = mlflow.get_run(run_id)
    return run.data.metrics.get("mean_score")


def get_best_improved_run_id(
    experiment_name: str,
    direction: str,
    branch: str,
) -> str | None:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.status = 'improved' AND tags.branch = '{branch}'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    if runs.empty or "metrics.mean_score" not in runs.columns:
        return None

    runs = runs.dropna(subset=["metrics.mean_score"])
    if runs.empty:
        return None

    ascending = direction != "maximize"
    runs = runs.sort_values("metrics.mean_score", ascending=ascending)
    return runs.iloc[0]["run_id"]


def ensure_submissions_experiment(
    experiment_prefix: str,
    competition_slug: str,
    submissions_suffix: str = "submissions",
) -> tuple[str, str]:
    name = f"{experiment_prefix}_{competition_slug}_{submissions_suffix}"
    return name, ensure_experiment(name)
