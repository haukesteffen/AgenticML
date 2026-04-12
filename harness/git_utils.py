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


def solution_has_diff(cwd: Path | None = None) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "solution.py"],
        cwd=cwd,
        capture_output=True,
    )
    return result.returncode != 0


def read_hypothesis_via_ast(solution_path: Path) -> str:
    try:
        source = solution_path.read_text()
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "HYPOTHESIS":
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            return node.value.value
    except Exception:
        pass
    return "[missing hypothesis]"


def commit_allowlist(files: list[str], message: str, cwd: Path | None = None) -> str:
    existing = [f for f in files if (cwd / f).exists()] if cwd else files
    if not existing:
        raise RuntimeError("No files to commit")
    _run_git("add", *existing, cwd=cwd)
    _run_git("commit", "-m", message, cwd=cwd)
    return head_sha(cwd)


def reset_one(cwd: Path | None = None) -> None:
    _run_git("reset", "--hard", "HEAD~1", cwd=cwd)
