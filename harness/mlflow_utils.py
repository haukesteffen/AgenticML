from __future__ import annotations

import mlflow
from mlflow.entities import ViewType


def setup_autolog() -> None:
    mlflow.autolog(
        log_models=False,
        log_datasets=False,
        log_input_examples=False,
        log_model_signatures=False,
        silent=True,
    )


def ensure_experiment(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return mlflow.create_experiment(experiment_name)
    return experiment.experiment_id


def start_parent_run(
    experiment_name: str,
    tags: dict[str, str],
) -> mlflow.ActiveRun:
    experiment_id = ensure_experiment(experiment_name)
    run = mlflow.start_run(experiment_id=experiment_id, tags=tags)
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
