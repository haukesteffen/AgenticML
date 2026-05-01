from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def _run_git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def current_branch(cwd: Path | None = None) -> str:
    return _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)


def head_sha(cwd: Path | None = None) -> str:
    return _run_git("rev-parse", "HEAD", cwd=cwd)


def file_has_diff(path: str, cwd: Path | None = None) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", path],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def solution_has_diff(cwd: Path | None = None) -> bool:
    return file_has_diff("solution.py", cwd=cwd)


def file_exists_in_head(path: str, cwd: Path | None = None) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{path}"],
        cwd=cwd,
        capture_output=True,
    )
    return result.returncode == 0


def read_string_constant_via_ast(path: Path, name: str, default: str) -> str:
    try:
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            return node.value.value
    except Exception:
        pass
    return default


def read_hypothesis_via_ast(solution_path: Path) -> str:
    return read_string_constant_via_ast(solution_path, "HYPOTHESIS", "[missing hypothesis]")


def commit_allowlist(files: list[str], message: str, cwd: Path | None = None) -> str:
    existing = [f for f in files if (cwd / f).exists()] if cwd else files
    if not existing:
        raise RuntimeError("No files to commit")
    _run_git("add", *existing, cwd=cwd)
    _run_git("commit", "-m", message, cwd=cwd)
    return head_sha(cwd)


def reset_one(files: list[str], cwd: Path | None = None) -> None:
    _run_git("reset", "--soft", "HEAD~1", cwd=cwd)

    for path in files:
        if file_exists_in_head(path, cwd=cwd):
            _run_git("restore", "--source=HEAD", "--staged", "--worktree", "--", path, cwd=cwd)
            continue

        subprocess.run(
            ["git", "rm", "--cached", "-f", "--", path],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        file_path = (cwd / path) if cwd else Path(path)
        if file_path.exists():
            file_path.unlink()
