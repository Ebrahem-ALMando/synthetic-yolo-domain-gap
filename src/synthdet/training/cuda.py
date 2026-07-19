"""CUDA verification and one-time common-profile selection for Sprint 4B."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthdet.synthetic.contracts import sha256_file, stable_json_hash

MINIMUM_DISK_BYTES = 10 * 1024**3
MINIMUM_FREE_VRAM_BYTES = 512 * 1024**2


def inspect_cuda(device: str, storage_root: Path) -> dict[str, Any]:
    import torch
    import ultralytics

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("PyTorch CUDA is unavailable; final training is prohibited")
    try:
        device_index = int(str(device).split(":")[-1])
    except ValueError as error:
        raise ValueError(f"CUDA device must be an integer index, got {device!r}") from error
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_index} is not visible")
    torch_device = torch.device(f"cuda:{device_index}")
    left = torch.arange(256, dtype=torch.float32, device=torch_device).reshape(16, 16)
    result = left @ left.T
    torch.cuda.synchronize(torch_device)
    if result.device.type != "cuda" or not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("CUDA tensor-operation verification failed")
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch_device)
    disk = shutil.disk_usage(storage_root)
    if disk.free < MINIMUM_DISK_BYTES:
        raise RuntimeError(
            f"Insufficient free disk: {disk.free} bytes; require {MINIMUM_DISK_BYTES}"
        )
    nvidia_smi = None
    executable = shutil.which("nvidia-smi")
    if executable:
        command = [
            executable,
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
            f"--id={device_index}",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        nvidia_smi = (
            completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()
        )
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "gpu_model": torch.cuda.get_device_name(device_index),
        "gpu_index": device_index,
        "vram_total_bytes": total_bytes,
        "vram_free_bytes_before_preflight": free_bytes,
        "disk_free_bytes": disk.free,
        "cuda_tensor_operation_verified": True,
        "ultralytics_device_detection_verified": True,
        "nvidia_smi": nvidia_smi,
        "software_versions": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
        },
        "colab": {
            "detected": "COLAB_RELEASE_TAG" in os.environ,
            "release_tag": os.environ.get("COLAB_RELEASE_TAG"),
            "python_executable": sys.executable,
        },
    }


def select_common_profile(
    run_attempt: Callable[[str, int], dict[str, Any]], total_vram_bytes: int
) -> tuple[str, list[dict[str, Any]]]:
    """Try batch 16, then batch 4; freeze exactly one common profile."""

    attempts: list[dict[str, Any]] = []
    try:
        standard = run_attempt("standard", 16)
        attempts.append(standard)
        margin = int(standard["free_vram_bytes_after"])
        required = max(MINIMUM_FREE_VRAM_BYTES, int(total_vram_bytes * 0.10))
        if standard.get("status") == "passed" and margin >= required:
            return "standard", attempts
        standard["selection_rejection"] = (
            f"free VRAM safety margin {margin} is below required {required}"
        )
    except RuntimeError as error:
        attempts.append(
            {"profile_name": "standard", "batch": 16, "status": "failed", "error": str(error)}
        )
    try:
        low_memory = run_attempt("low_memory", 4)
        attempts.append(low_memory)
        if low_memory.get("status") == "passed":
            return "low_memory", attempts
    except RuntimeError as error:
        attempts.append(
            {"profile_name": "low_memory", "batch": 4, "status": "failed", "error": str(error)}
        )
    raise RuntimeError("Both frozen CUDA batch profiles failed; use a GPU with more available VRAM")


def freeze_profile(
    profile_name: str,
    attempts: list[dict[str, Any]],
    environment: dict[str, Any],
    common: dict[str, Any],
    base_weight_path: Path,
    device: str,
) -> dict[str, Any]:
    selected = common["hardware_profiles"][profile_name]
    identity_inputs = {
        "profile_name": profile_name,
        "batch": int(selected["batch"]),
        "imgsz": int(selected["imgsz"]),
        "device": str(device),
        "base_weight_sha256": sha256_file(base_weight_path),
        "identities": common["identities"],
        "preflight": attempts,
    }
    return {
        "status": "frozen",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": environment,
        **identity_inputs,
        "profile_identity": stable_json_hash(identity_inputs),
        "test_set_access_count": 0,
        "scientific_result": False,
    }


def write_profile(profile: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("profile_identity") != profile["profile_identity"]:
            raise FileExistsError(f"Refusing to replace a different frozen profile: {output}")
        return
    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
