from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow

from harness import git_utils, lock, mlflow_utils
from harness.config import HarnessConfig

EXIT_INVALID_OUTPUT = 2
EXIT_FOLD_TIMEOUT = 3
GRACE_SECONDS = 5
SETUP_BUFFER_SECONDS = 120
RESET_ALLOWLIST = ("NOTES.md",)


@dataclass(frozen=True)
class ExperimentSpec:
    kind: str
    file_name: str
    smoke_worker: str
    full_worker: str


MODEL_EXPERIMENT = ExperimentSpec(
    kind="model",
    file_name="solution.py",
    smoke_worker="harness.worker_smoke",
    full_worker="harness.worker_full",
)

ENSEMBLE_EXPERIMENT = ExperimentSpec(
    kind="ensemble",
    file_name="ensemble.py",
    smoke_worker="harness.worker_ensemble_smoke",
    full_worker="harness.worker_ensemble_full",
)


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


def _classify_worker_exit(returncode: int) -> str:
    if returncode == 0:
        return "ok"
    if returncode == EXIT_INVALID_OUTPUT:
        return "invalid_output"
    if returncode == EXIT_FOLD_TIMEOUT:
        return "fold_timeout"
    if returncode == -1:
        return "timeout"
    return "crash"


def run(config_path: str = "config.yaml") -> None:
    cfg = HarnessConfig.load(config_path)
    project_root = cfg.project_root
    config_abs = str((Path(config_path)).resolve())

    lock.acquire(project_root)
    try:
        _run_inner(cfg, project_root, config_abs)
    finally:
        lock.release(project_root)


def _run_inner(cfg: HarnessConfig, project_root: Path, config_abs: str) -> None:
    spec = _detect_experiment(project_root)
    if spec is None:
        print("Nothing to do — solution.py and ensemble.py have no pending changes.")
        return

    experiment_path = project_root / spec.file_name
    hypothesis = git_utils.read_hypothesis_via_ast(experiment_path)

    sha = git_utils.commit_allowlist(
        [spec.file_name, "NOTES.md"],
        message=hypothesis,
        cwd=project_root,
    )
    branch = git_utils.current_branch(cwd=project_root)
    slug = cfg.mlflow.competition_slug

    experiment_name = f"{cfg.mlflow.experiment_prefix}_{slug}_{branch}"
    parent = mlflow_utils.start_parent_run(
        experiment_name,
        tags={
            "branch": branch,
            "sha": sha,
            "hypothesis": hypothesis,
            "experiment_kind": spec.kind,
            "status": "running",
        },
    )
    parent_run_id = parent.info.run_id

    try:
        _run_with_parent(cfg, project_root, config_abs, branch, parent_run_id, spec)
    except Exception:
        mlflow_utils.end_parent_run("error")
        git_utils.reset_one([spec.file_name, *RESET_ALLOWLIST], cwd=project_root)
        raise


def _detect_experiment(project_root: Path) -> ExperimentSpec | None:
    solution_changed = git_utils.file_has_diff("solution.py", cwd=project_root)
    ensemble_path = project_root / "ensemble.py"
    ensemble_changed = ensemble_path.exists() and git_utils.file_has_diff("ensemble.py", cwd=project_root)

    if solution_changed and ensemble_changed:
        raise RuntimeError(
            "Both solution.py and ensemble.py changed. Edit exactly one experiment file per run."
        )
    if solution_changed:
        return MODEL_EXPERIMENT
    if ensemble_changed:
        return ENSEMBLE_EXPERIMENT
    return None


def _module_loc(project_root: Path, file_name: str) -> int:
    try:
        return len((project_root / file_name).read_text().splitlines())
    except OSError:
        return 0


def _smoke_timeout(cfg: HarnessConfig, spec: ExperimentSpec) -> int:
    if spec.kind == "ensemble":
        return cfg.budget.smoke_seconds + SETUP_BUFFER_SECONDS
    return cfg.budget.smoke_seconds


def _should_run_smoke(spec: ExperimentSpec) -> bool:
    return spec.kind != "ensemble"


def _print_result(status: str, score, best, n_folds: int, elapsed: float, loc: int) -> None:
    score_s = f"{score:.6f}" if isinstance(score, float) else "-"
    best_s = f"{best:.6f}" if isinstance(best, float) else "-"
    print(
        f"RESULT status={status} score={score_s} best={best_s}"
        f" folds={n_folds} elapsed={int(elapsed)}s loc={loc}"
    )


def _run_with_parent(
    cfg: HarnessConfig,
    project_root: Path,
    config_abs: str,
    branch: str,
    parent_run_id: str,
    spec: ExperimentSpec,
) -> None:
    t0 = time.monotonic()
    loc = _module_loc(project_root, spec.file_name)
    slug = cfg.mlflow.competition_slug

    # --- smoke phase ---
    if _should_run_smoke(spec):
        print("Running smoke test...")
        smoke_code, smoke_stderr = _spawn_worker(
            spec.smoke_worker,
            config_abs,
            timeout=_smoke_timeout(cfg, spec),
        )
        smoke_status = _classify_worker_exit(smoke_code)

        if smoke_status != "ok":
            print(f"Smoke test failed: {smoke_status}")
            if smoke_stderr:
                print(smoke_stderr)
            status = f"smoke_fail:{smoke_status}"
            mlflow_utils.log_traceback_artifact(smoke_stderr, "smoke_traceback.txt")
            mlflow_utils.end_parent_run(status)
            git_utils.reset_one([spec.file_name, *RESET_ALLOWLIST], cwd=project_root)
            _print_result(status, None, None, 0, time.monotonic() - t0, loc)
            return

        print("Smoke test passed. Running full CV...")
    else:
        print("Skipping smoke test for ensemble experiment. Running full CV...")

    # --- full phase ---
    full_timeout = cfg.budget.fold_seconds * cfg.cv.n_splits + SETUP_BUFFER_SECONDS
    full_code, full_stderr = _spawn_worker(
        spec.full_worker,
        config_abs,
        timeout=full_timeout,
        extra_env={
            "MLFLOW_RUN_ID": parent_run_id,
            "HARNESS_BRANCH": branch,
            "HARNESS_SLUG": slug,
        },
    )
    full_status = _classify_worker_exit(full_code)

    if full_status != "ok":
        print(f"Full run failed: {full_status}")
        if full_stderr:
            print(full_stderr)
        status = f"fail:{full_status}"
        mlflow_utils.log_traceback_artifact(full_stderr, "full_traceback.txt")
        mlflow_utils.end_parent_run(status)
        git_utils.reset_one([spec.file_name, *RESET_ALLOWLIST], cwd=project_root)
        _print_result(status, None, None, 0, time.monotonic() - t0, loc)
        return

    # --- classify result ---
    mean_score = mlflow_utils.get_parent_mean_score(parent_run_id)
    if mean_score is None:
        print("Could not read mean_score from MLflow run.")
        status = "fail:no_score"
        mlflow_utils.end_parent_run(status)
        git_utils.reset_one([spec.file_name, *RESET_ALLOWLIST], cwd=project_root)
        _print_result(status, None, None, cfg.cv.n_splits, time.monotonic() - t0, loc)
        return

    experiment_name = f"{cfg.mlflow.experiment_prefix}_{slug}_{branch}"
    best_score = mlflow_utils.get_best_score(experiment_name, cfg.metric.direction, branch)

    if best_score is None:
        improved = True
    elif cfg.metric.direction == "maximize":
        improved = mean_score > best_score
    else:
        improved = mean_score < best_score

    # Log the experiment file snapshot as an artifact on the parent run.
    client = mlflow.tracking.MlflowClient()
    client.log_artifact(parent_run_id, str(project_root / spec.file_name))

    if improved:
        mlflow_utils.end_parent_run("improved")
        print(f"Improved! mean_score={mean_score:.6f} (previous best: {best_score})")
        _print_result("improved", mean_score, best_score, cfg.cv.n_splits, time.monotonic() - t0, loc)
    else:
        mlflow_utils.end_parent_run("regressed")
        git_utils.reset_one([spec.file_name, *RESET_ALLOWLIST], cwd=project_root)
        print(f"Regressed. mean_score={mean_score:.6f} vs best={best_score:.6f} — branch reset.")
        _print_result("regressed", mean_score, best_score, cfg.cv.n_splits, time.monotonic() - t0, loc)
