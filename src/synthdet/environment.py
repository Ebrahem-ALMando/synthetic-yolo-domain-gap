"""Command-line checks for a usable Sprint 1 development environment."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from synthdet.config import load_config, validate_project_directories
from synthdet.config.loader import DEFAULT_CONFIG_PATH, PROJECT_ROOT

MINIMUM_PYTHON = (3, 11)


def _cuda_status() -> str:
    if importlib.util.find_spec("torch") is None:
        return "not checked (PyTorch is not installed); CPU-only setup is supported"
    try:
        import torch

        if torch.cuda.is_available():
            return f"available ({torch.cuda.get_device_name(0)})"
        return "unavailable; CPU-only setup is supported"
    except (ImportError, RuntimeError) as error:
        return f"not checked ({error}); CPU-only setup is supported"


def main() -> int:
    """Run non-destructive environment checks and return a shell status code."""

    failures: list[str] = []
    version = sys.version_info[:3]
    version_text = ".".join(str(part) for part in version)
    if version < MINIMUM_PYTHON:
        failures.append(f"Python {version_text} is below the required 3.11")
    else:
        print(f"[OK] Python {version_text}")

    config_path = Path(os.getenv("SYNTHDET_CONFIG", str(DEFAULT_CONFIG_PATH)))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    try:
        config = load_config(config_path)
        print(f"[OK] Configuration loaded: {config.project.name} (seed={config.seed})")
    except (FileNotFoundError, OSError, ValueError, ValidationError) as error:
        failures.append(f"Configuration error: {error}")

    missing = validate_project_directories()
    if missing:
        failures.append("Missing directories: " + ", ".join(str(path) for path in missing))
    else:
        print("[OK] Required project directories exist")

    print(f"[INFO] CUDA: {_cuda_status()}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1
    print("[OK] Environment foundation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

