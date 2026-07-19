"""Training hardware and software environment reporting."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _powershell_json(command: str) -> Any:
    if os.name != "nt":
        return None
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def collect_environment(project_root: Path) -> dict[str, Any]:
    cpu = _powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfLogicalProcessors "
        "| ConvertTo-Json -Compress"
    ) or {}
    video = _powershell_json(
        "@(Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion) "
        "| ConvertTo-Json -Compress"
    ) or []
    if isinstance(video, dict):
        video = [video]
    nvidia_smi: list[dict[str, Any]] = []
    nvidia_command = shutil.which("nvidia-smi")
    if nvidia_command:
        result = subprocess.run(
            [
                nvidia_command,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                values = [value.strip() for value in line.split(",")]
                if len(values) == 4:
                    nvidia_smi.append(
                        {
                            "name": values[0],
                            "memory_total_mib": int(values[1]),
                            "memory_free_mib": int(values[2]),
                            "driver_version": values[3],
                        }
                    )
    torch_version = _package_version("torch")
    ultralytics_version = _package_version("ultralytics")
    cuda_available = False
    cuda_runtime = None
    torch_devices: list[dict[str, Any]] = []
    if torch_version:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_runtime = torch.version.cuda
        if cuda_available:
            torch_devices = [
                {
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(project_root)
    if torch_version is None or ultralytics_version is None:
        classification = "blocked_missing_runtime"
    elif memory.total < 4 * 1024**3 or disk.free < 5 * 1024**3:
        classification = "blocked_insufficient_resources"
    elif cuda_available and any(device["memory_bytes"] >= 6 * 1024**3 for device in torch_devices):
        classification = "full_training_ready_gpu"
    else:
        classification = "smoke_training_only_cpu"
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "classification": classification,
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": cpu.get("Name") or platform.processor() or "unknown",
        "logical_core_count": int(cpu.get("NumberOfLogicalProcessors") or os.cpu_count() or 0),
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "virtual_environment": {
            "active": sys.prefix != sys.base_prefix,
            "prefix": str(Path(sys.prefix).resolve()),
        },
        "torch_version": torch_version,
        "ultralytics_version": ultralytics_version,
        "cuda_available": cuda_available,
        "cuda_runtime_version": cuda_runtime,
        "torch_cuda_devices": torch_devices,
        "display_adapters": video,
        "nvidia_smi": nvidia_smi,
    }


def write_environment_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gib = 1024**3
    adapters = ", ".join(
        f"{item.get('Name', 'unknown')} "
        f"({int(item.get('AdapterRAM') or 0) / gib:.2f} GiB adapter memory)"
        for item in report["display_adapters"]
    )
    nvidia = ", ".join(
        f"{item['name']} ({item['memory_total_mib']} MiB total, "
        f"{item['memory_free_mib']} MiB free, driver {item['driver_version']})"
        for item in report["nvidia_smi"]
    )
    lines = [
        "# Sprint 4A Training Environment",
        "",
        f"- Classification: `{report['classification']}`",
        f"- OS: {report['operating_system']}",
        f"- Python: {report['python_version']}",
        f"- CPU: {report['cpu_model']} ({report['logical_core_count']} logical cores)",
        f"- RAM: {report['ram_total_bytes'] / gib:.2f} GiB total; "
        f"{report['ram_available_bytes'] / gib:.2f} GiB available at inspection",
        f"- Project-volume free space: {report['disk_free_bytes'] / gib:.2f} GiB",
        f"- PyTorch: {report['torch_version'] or 'not installed'}",
        f"- Ultralytics: {report['ultralytics_version'] or 'not installed'}",
        f"- CUDA detected by PyTorch: {report['cuda_available']}",
        f"- CUDA runtime: {report['cuda_runtime_version'] or 'none'}",
        f"- Display adapters: {adapters or 'not detected'}",
        f"- NVIDIA driver report: {nvidia or 'not available'}",
        f"- Virtual environment active: {report['virtual_environment']['active']}",
        "",
        "This classification is based on PyTorch detection, not merely the presence of a GPU "
        "driver.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
