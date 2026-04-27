from __future__ import annotations

import platform
import shutil
from typing import Any


def detect_ncu() -> dict[str, Any]:
    ncu_path = shutil.which("ncu")
    if ncu_path:
        return {
            "available": True,
            "path": ncu_path,
            "enabled": False,
            "status": "detected_not_used_in_mvp",
        }
    return {
        "available": False,
        "path": None,
        "enabled": False,
        "status": "not_detected",
    }


def runtime_environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
