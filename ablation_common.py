"""Shared subprocess/fingerprinting primitives used by the ablation entry points.

``ablation_runner.py`` and ``run_method_comparison.py`` (and ``run_checkpoint_sweep.py``)
each drive subprocesses over a teacher checkpoint and need the same handful of
low-level building blocks. They used to carry independent copies of these that
drifted apart; this module is the single source.
"""

from __future__ import annotations

from datetime import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess


def file_fingerprint(path: str | Path) -> dict:
    """Return the sha256/size identity of a file, or an ``exists: False`` stub."""
    path = Path(path).resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def content_identity(fingerprint: dict) -> dict:
    """Return a fingerprint's identity without its on-disk location."""
    if fingerprint.get("sha256"):
        return {"size": fingerprint.get("size"), "sha256": fingerprint["sha256"]}
    return {"exists": False, "path": fingerprint.get("path")}


def acquire_run_lock(path: Path):
    """Take an exclusive, non-blocking lock on ``path``, recording the owner pid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown process"
        handle.close()
        raise RuntimeError(f"Another ablation is already using this catalog entry ({owner})") from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={datetime.now().isoformat()}\n")
    handle.flush()
    return handle


def run_live_subprocess(command: list[str], log_path: Path) -> int:
    """Run ``command``, streaming its combined stdout/stderr to the console and to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1
        )
        assert process.stdout is not None
        for line in process.stdout:
            print("  | " + line, end="", flush=True)
            stream.write(line)
        return process.wait()


def default_estimator_window(task_id: str) -> int:
    """Default DAgger history window for a Gym task id.

    Mirrors ``train_state_estimator.py``'s own ``--window`` default. Duplicated
    here rather than imported because that module launches the Isaac Sim app
    as an import-time side effect and cannot be imported just for this constant.
    """
    return 25 if task_id == "Isaac-G1-AMP-Walk-JOSE-Direct-v0" else 50


def implementation_fingerprint(paths: tuple[str, ...]) -> str:
    """SHA-256 over the raw bytes of the source files that define a job's behaviour.

    The path name is mixed in alongside the bytes so that reordering or renaming
    the list is itself a change. Hashing bytes rather than an AST means a
    comment-only edit invalidates cached results too -- deliberately
    conservative: a stale result silently reused is far more expensive than a
    re-run.
    """
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for relative in paths:
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()
