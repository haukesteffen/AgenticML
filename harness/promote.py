"""Promotion step — run nested 5x5 on the current solution.py and log honest
artifacts to the ``promoted`` MLflow experiment.

No improvement gate. No git commit. The lane's source branch and head SHA are
captured as tags so phase e can attribute promoted runs back to their origin.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import tempfile
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from harness import git_utils, mlflow_utils
from harness.config import HarnessConfig
from harness.metric import get_metric
from harness.nested_cv import nested_oof
from harness.worker_smoke import validate_predictions


def _promoted_experiment_name(cfg: HarnessConfig) -> str:
    return f"{cfg.mlflow.experiment_prefix}_{cfg.mlflow.competition_slug}_promoted"


def _resolve_family(solution_path: Path) -> str:
    return git_utils.read_string_constant_via_ast(solution_path, "FAMILY", "unknown")


def main(config_path: str = "config.yaml") -> None:
    cfg = HarnessConfig.load(config_path)
    project_root = cfg.project_root
    solution_path = project_root / "solution.py"

    hypothesis = git_utils.read_hypothesis_via_ast(solution_path)
    recipe = git_utils.read_string_constant_via_ast(solution_path, "RECIPE", "v1_raw")
    family = _resolve_family(solution_path)
    if family == "unknown":
        print(
            "warn: solution.py has no FAMILY constant — promoted run will be tagged family=unknown.",
            file=sys.stderr,
        )

    branch = git_utils.current_branch(cwd=project_root)
    sha = git_utils.head_sha(cwd=project_root)
    lane = f"{recipe}__{family}"

    train_df = pd.read_csv(project_root / cfg.dataset.train_path)
    test_df = pd.read_csv(project_root / cfg.dataset.test_path)

    sys.path.insert(0, str(project_root))
    import solution  # noqa: E402
    from features import load_recipe  # noqa: E402

    X, X_test = load_recipe(
        recipe, train_df, test_df,
        target=cfg.dataset.target,
        id_column=cfg.dataset.id_column,
        project_root=project_root,
    )
    y_raw = train_df[cfg.dataset.target]

    if cfg.dataset.problem_type != "regression":
        y, classes = pd.factorize(y_raw, sort=True)
        n_classes = len(classes)
    else:
        y = y_raw.values.astype(float)
        n_classes = 0

    metric_fn, _ = get_metric(cfg.metric.name)

    experiment_name = _promoted_experiment_name(cfg)
    mlflow_utils.ensure_experiment(experiment_name)
    run = mlflow.start_run(
        tags={
            "lane": lane,
            "recipe": recipe,
            "family": family,
            "hypothesis": hypothesis,
            "source_branch": branch,
            "source_commit": sha,
            "promoted_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "status": "running",
        },
    )
    run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()

    try:
        outer_oof, inner_oof = nested_oof(
            solution.fit_predict,
            X, y,
            problem_type=cfg.dataset.problem_type,
            cv_config=cfg.cv,
            n_classes=n_classes,
        )

        validate_predictions(outer_oof, len(y), n_classes, cfg.dataset.problem_type)
        outer_score = float(metric_fn(y, outer_oof, cfg.dataset.problem_type))
        client.log_metric(run_id, "cv_score_outer", outer_score)

        # Final retrain on 100% for the test prediction.
        test_pred = np.asarray(solution.fit_predict(X, y, X_test))
        validate_predictions(test_pred, len(X_test), n_classes, cfg.dataset.problem_type)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            np.save(tmp / "oof_outer.npy", outer_oof)
            np.save(tmp / "oof_inner.npy", inner_oof)
            np.save(tmp / "test_pred.npy", test_pred)
            for f in ("oof_outer.npy", "oof_inner.npy", "test_pred.npy"):
                client.log_artifact(run_id, str(tmp / f))

        client.set_tag(run_id, "status", "promoted")
        mlflow.end_run()
        print(f"PROMOTED lane={lane} score={outer_score:.6f} run_id={run_id}")
    except Exception:
        traceback.print_exc()
        try:
            client.set_tag(run_id, "status", "error")
        finally:
            mlflow.end_run("FAILED")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
