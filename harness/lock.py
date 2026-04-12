from __future__ import annotations

import os
from pathlib import Path

LOCK_FILE = ".harness.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def acquire(project_root: Path) -> Path:
    lock_path = project_root / LOCK_FILE
    if lock_path.exists():
        try:
            stored_pid = int(lock_path.read_text().strip())
        except ValueError:
            stored_pid = -1
        if _pid_alive(stored_pid):
            raise RuntimeError(
                f"Another harness run is active (PID {stored_pid}). "
                f"Remove {lock_path} manually if that process is dead."
            )
        lock_path.unlink()

    lock_path.write_text(str(os.getpid()))
    return lock_path


def release(project_root: Path) -> None:
    lock_path = project_root / LOCK_FILE
    if lock_path.exists():
        lock_path.unlink()
