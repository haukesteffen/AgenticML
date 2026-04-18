"""Manual Kaggle submission from a logged CV run.

Flow:
  1. Resolve source CV run (best improved on current branch, or explicit --run-id).
  2. Download its solution.py artifact.
  3. Spawn worker_submit to refit on full train and predict test.
  4. Upload submission.csv to Kaggle and poll for the public LB score.
  5. Log everything (solution.py, submission.csv, test_predictions.npy, metrics)
     to a dedicated `{prefix}_{slug}_submissions` mlflow experiment.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow

from harness import git_utils, lock, mlflow_utils
from harness.config import HarnessConfig

EXIT_INVALID_OUTPUT = 2
EXIT_REFIT_TIMEOUT = 3
GRACE_SECONDS = 5
SETUP_BUFFER_SECONDS = 120


def submit(
    config_path: str = "config.yaml",
    run_id: str | None = None,
    message: str | None = None,
    branch: str | None = None,
) -> None:
    cfg = HarnessConfig.load(config_path)
    project_root = cfg.project_root
    config_abs = str(Path(config_path).resolve())

    lock.acquire(project_root)
    try:
        _submit_inner(cfg, project_root, config_abs, run_id, message, branch)
    finally:
        lock.release(project_root)


def _submit_inner(
    cfg: HarnessConfig,
    project_root: Path,
    config_abs: str,
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
        downloaded = client.download_artifacts(run_id, "solution.py", str(artifact_dir))
        solution_path = Path(downloaded)

        out_dir = tmpdir / "out"
        out_dir.mkdir()

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
            _run_submission(
                cfg=cfg,
                project_root=project_root,
                config_abs=config_abs,
                solution_path=solution_path,
                out_dir=out_dir,
                parent_run_id=parent_run_id,
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


def _run_submission(
    cfg: HarnessConfig,
    project_root: Path,
    config_abs: str,
    solution_path: Path,
    out_dir: Path,
    parent_run_id: str,
    source_branch: str,
    source_sha: str,
    hypothesis: str,
    cv_mean: float | None,
    cv_std: float | None,
    message: str | None,
) -> None:
    client = mlflow.tracking.MlflowClient()

    print("Refitting on full train and predicting test...")
    refit_timeout = cfg.budget.fold_seconds * cfg.cv.n_splits + SETUP_BUFFER_SECONDS
    code, stderr = _spawn_worker(
        "harness.worker_submit",
        config_abs,
        timeout=refit_timeout,
        extra_env={
            "HARNESS_SOLUTION_PATH": str(solution_path),
            "HARNESS_OUT_DIR": str(out_dir),
        },
    )
    if code != 0:
        print(f"Refit failed (exit={code}).")
        if stderr:
            print(stderr)
        mlflow_utils.log_traceback_artifact(stderr, "refit_traceback.txt")
        mlflow_utils.end_parent_run("refit_failed")
        return

    submission_csv = out_dir / "submission.csv"
    predictions_npy = out_dir / "test_predictions.npy"

    client.log_artifact(parent_run_id, str(solution_path))
    client.log_artifact(parent_run_id, str(submission_csv))
    client.log_artifact(parent_run_id, str(predictions_npy))
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


def _spawn_worker(
    module: str,
    config_path: str,
    timeout: int,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    env = {**os.environ, **(extra_env or {})}
    proc = subprocess.Popen(
        [sys.executable, "-m", module, "--config", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stderr.decode(errors="replace")
    except subprocess.TimeoutExpired:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait()
        return -1, "Killed: wall-clock timeout exceeded"
