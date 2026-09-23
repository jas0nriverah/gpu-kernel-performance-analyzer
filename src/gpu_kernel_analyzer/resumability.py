"""Local transactional checkpoints for benchmark sweeps."""
from __future__ import annotations

import hashlib
import json
import os
import time

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

CHECKPOINT_NAME = ".sweep_checkpoint.json"


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


@contextmanager
def sweep_lock(run_dir: Path) -> Iterator[None]:
    """Serialize local sweeps; flock is released automatically after crashes."""
    if fcntl is None:
        raise RuntimeError("Resumable sweeps require POSIX advisory file locks")
    lock = run_dir.with_name(f".{run_dir.name}.sweep.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another sweep is using this output directory: {run_dir}") from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} started={time.time()}\n".encode())
        os.fsync(fd)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def new_checkpoint(identity: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {"format_version": 1, "identity": identity, "run_id": run_id, "completed": [], "checkpoint_sha256": ""}


def seal(checkpoint: dict[str, Any]) -> dict[str, Any]:
    result = dict(checkpoint)
    result["checkpoint_sha256"] = digest({k: v for k, v in result.items() if k != "checkpoint_sha256"})
    return result


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    atomic_json(path, seal(checkpoint))


def load_checkpoint(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read sweep checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported or malformed sweep checkpoint")
    expected = checkpoint.get("checkpoint_sha256")
    actual = digest({k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"})
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("Sweep checkpoint integrity check failed; checkpoint may have been tampered with")
    if checkpoint.get("identity") != identity:
        raise ValueError("Sweep checkpoint is incompatible with the current binary, scenarios, or protocol configuration")
    completed = checkpoint.get("completed")
    if not isinstance(completed, list):
        raise ValueError("Malformed completed-scenario list in sweep checkpoint")
    seen: set[int] = set()
    for entry in completed:
        if not isinstance(entry, dict) or not isinstance(entry.get("index"), int) or not isinstance(entry.get("raw"), dict):
            raise ValueError("Malformed completed scenario in sweep checkpoint")
        idx = entry["index"]
        if idx in seen or idx < 0 or entry.get("raw_sha256") != digest(entry["raw"]):
            raise ValueError("Sweep checkpoint scenario integrity check failed")
        seen.add(idx)
    return checkpoint


def current_device_identity(device: dict[str, Any]) -> str | None:
    """Return a stable visible-device identity where the platform exposes one."""
    uuid = device.get("uuid")
    if isinstance(uuid, str) and uuid.strip():
        return "cuda-uuid:" + uuid.strip().lower()
    name = str(device.get("name", ""))
    if name.lower().startswith("fake"):
        return "fixture:" + digest(device)
    return None


def probe_live_device_identity(binary: Path, interpreter: str | None, timeout_seconds: float) -> str | None:
    """Ask the worker for the UUID of the device selected by its CUDA runtime."""
    import subprocess

    command = ([interpreter] if interpreter else []) + [str(binary), "--device-info"]
    try:
        proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=min(timeout_seconds, 15.0))
        if proc.returncode != 0:
            return None
        payload = json.loads([line for line in proc.stdout.splitlines() if line.strip()][-1])
        device = payload.get("device", {}) if isinstance(payload, dict) else {}
        if not isinstance(device, dict):
            return None
        return current_device_identity(device)
    except Exception:
        return None
