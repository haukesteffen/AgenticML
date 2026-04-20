"""Manual Kaggle submission from a logged CV run.

Flow:
  1. Resolve source CV run (best improved on current branch, or explicit --run-id).
  2. Download its test_predictions.npy + predictions_manifest.json artifacts.
  3. Map predictions to submission labels via the manifest's class_values.
  4. Upload submission.csv to Kaggle and poll for the public LB score.
  5. Log a slim record (submission.csv + lb metric + source_run_id tag) to
     the `{prefix}_{slug}_submissions` mlflow experiment.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from harness import git_utils, lock, mlflow_utils
from harness.config import HarnessConfig


def submit(
    config_path: str = "config.yaml",
    run_id: str | None = None,
    message: str | None = None,
    branch: str | None = None,
) -> None:
    cfg = HarnessConfig.load(config_path)
    project_root = cfg.project_root

    lock.acquire(project_root)
    try:
        _submit_inner(cfg, project_root, run_id, message, branch)
    finally:
        lock.release(project_root)


def _submit_inner(
    cfg: HarnessConfig,
    project_root: Path,
    run_id: str | None,
    message: str | None,
    branch_override: str | None,
) -> None:
    branch = branch_override or git_utils.current_branch(cwd=project_root)
    source_experiment = f"{cfg.mlflow.experiment_prefix}_{cfg.mlflow.competition_slug}_{branch}"

    if run_id is None:
        run_id = mlflow_utils.get_best_improved_run_id(
            source_experiment, cfg.metric.direction, branch
        )
        if run_id is None:
            raise RuntimeError(
                f"No improved runs found in experiment {source_experiment!r} for branch {branch!r}. "
                f"Pass --run-id explicitly or run an experiment first."
            )
        print(f"Source run: {run_id} (best improved on {branch})")
    else:
        print(f"Source run: {run_id} (explicit)")

    client = mlflow.tracking.MlflowClient()
    source_run = client.get_run(run_id)
    source_tags = source_run.data.tags
    source_metrics = source_run.data.metrics
    experiment_kind = source_tags.get("experiment_kind", "model")
    if experiment_kind != "model":
        raise RuntimeError(
            "Submission for ensemble runs lands in the next rollout. "
            "Base-model runs now materialize test_predictions.npy at CV time; "
            "ensemble runs will follow."
        )

    source_branch = source_tags.get("branch", branch)
    source_sha = source_tags.get("sha", "")
    hypothesis = source_tags.get("hypothesis", "[missing hypothesis]")
    cv_mean = source_metrics.get("mean_score")
    cv_std = source_metrics.get("std_score")

    submissions_experiment, _ = mlflow_utils.ensure_submissions_experiment(
        cfg.mlflow.experiment_prefix,
        cfg.mlflow.competition_slug,
        cfg.mlflow.submissions_suffix,
    )

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        artifact_dir = tmpdir / "source_artifacts"
        artifact_dir.mkdir()

        test_preds_path = Path(
            client.download_artifacts(run_id, "test_predictions.npy", str(artifact_dir))
        )
        manifest_path = Path(
            client.download_artifacts(run_id, "predictions_manifest.json", str(artifact_dir))
        )
        test_preds = np.load(str(test_preds_path))
        manifest = json.loads(manifest_path.read_text())

        submission_csv = tmpdir / "submission.csv"
        _write_submission(cfg, project_root, test_preds, manifest, submission_csv)

        parent = mlflow_utils.start_parent_run(
            submissions_experiment,
            tags={
                "source_branch": source_branch,
                "source_run_id": run_id,
                "source_sha": source_sha,
                "source_experiment": source_experiment,
                "hypothesis": hypothesis,
                "status": "running",
            },
        )
        parent_run_id = parent.info.run_id

        try:
            _upload_and_log(
                cfg=cfg,
                client=client,
                parent_run_id=parent_run_id,
                submission_csv=submission_csv,
                source_branch=source_branch,
                source_sha=source_sha,
                hypothesis=hypothesis,
                cv_mean=cv_mean,
                cv_std=cv_std,
                message=message,
            )
        except Exception:
            mlflow_utils.end_parent_run("error")
            raise


def _write_submission(
    cfg: HarnessConfig,
    project_root: Path,
    test_preds: np.ndarray,
    manifest: dict,
    out_path: Path,
) -> None:
    test_df = pd.read_csv(project_root / cfg.dataset.test_path)
    sample_submission = pd.read_csv(project_root / "data" / "sample_submission.csv", nrows=0)

    test_ids = test_df[cfg.dataset.id_column]
    problem_type = manifest["problem_type"]
    class_values = manifest.get("class_values")

    if problem_type == "multiclass_classification":
        label_indices = np.argmax(test_preds, axis=1)
        labels = np.array(class_values)[label_indices]
    elif problem_type == "binary_classification":
        if test_preds.ndim == 2:
            label_indices = np.argmax(test_preds, axis=1)
        else:
            label_indices = (test_preds >= 0.5).astype(int)
        labels = np.array(class_values)[label_indices]
    else:
        labels = test_preds

    submission_cols = list(sample_submission.columns)
    submission = pd.DataFrame({
        submission_cols[0]: test_ids.values,
        submission_cols[1]: labels,
    })
    submission.to_csv(out_path, index=False)


def _upload_and_log(
    cfg: HarnessConfig,
    client: mlflow.tracking.MlflowClient,
    parent_run_id: str,
    submission_csv: Path,
    source_branch: str,
    source_sha: str,
    hypothesis: str,
    cv_mean: float | None,
    cv_std: float | None,
    message: str | None,
) -> None:
    client.log_artifact(parent_run_id, str(submission_csv))
    if cv_mean is not None:
        client.log_metric(parent_run_id, "cv_mean_score", cv_mean)
    if cv_std is not None:
        client.log_metric(parent_run_id, "cv_std_score", cv_std)

    final_message = message or _default_message(source_branch, source_sha, cv_mean, cv_std, hypothesis)
    print(f"Uploading to Kaggle: {final_message}")

    from harness import kaggle_utils

    try:
        ref = kaggle_utils.submit(cfg.mlflow.kaggle_slug, submission_csv, final_message)
    except Exception as e:
        print(f"Kaggle upload failed: {e}")
        client.set_tag(parent_run_id, "kaggle_error", str(e)[:500])
        mlflow_utils.end_parent_run("upload_failed")
        return

    client.set_tag(parent_run_id, "kaggle_filename", ref.file_name)
    client.set_tag(parent_run_id, "kaggle_description", ref.description)
    if ref.date:
        client.set_tag(parent_run_id, "kaggle_date", ref.date)

    print("Polling Kaggle for public LB score...")
    lb_public = kaggle_utils.poll_public_score(
        cfg.mlflow.kaggle_slug, ref, timeout=180, interval=15
    )

    if lb_public is not None:
        client.log_metric(parent_run_id, "lb_public", lb_public)
        if cv_mean is not None:
            client.log_metric(parent_run_id, "cv_lb_gap", cv_mean - lb_public)
        mlflow_utils.end_parent_run("submitted")
        print(f"SUBMITTED lb_public={lb_public:.6f} cv_mean={_fmt(cv_mean)} run={parent_run_id}")
    else:
        mlflow_utils.end_parent_run("pending_lb")
        print(f"SUBMITTED lb_public=pending run={parent_run_id}")


def _default_message(
    branch: str,
    sha: str,
    cv_mean: float | None,
    cv_std: float | None,
    hypothesis: str,
) -> str:
    short_sha = sha[:7] if sha else "nosha"
    cv_part = f"cv={cv_mean:.5f}" if cv_mean is not None else "cv=?"
    if cv_std is not None:
        cv_part += f"\u00b1{cv_std:.5f}"
    return f"{branch} | {short_sha} | {cv_part} | {hypothesis}"


def _fmt(x: float | None) -> str:
    return f"{x:.6f}" if isinstance(x, float) else "-"
