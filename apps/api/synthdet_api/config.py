"""Environment-backed API configuration with repository-portable defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Settings:
    repository_root: Path = REPOSITORY_ROOT
    model_root: Path | None = None
    device_policy: str = "auto"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    inference_timeout_seconds: int = 60
    max_batch_images: int = 4
    allowed_origins: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")

    @classmethod
    def from_environment(cls) -> Settings:
        device = os.getenv("SYNTHDET_DEVICE", "auto").lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("SYNTHDET_DEVICE must be auto, cpu, or cuda")
        raw_model_root = os.getenv("SYNTHDET_MODEL_ROOT")
        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "SYNTHDET_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
            ).split(",")
            if origin.strip()
        )
        if any(origin == "*" for origin in origins):
            raise ValueError("Wildcard CORS origins are not allowed")
        return cls(
            model_root=Path(raw_model_root).expanduser() if raw_model_root else None,
            device_policy=device,
            max_upload_bytes=_positive_int("SYNTHDET_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
            max_image_pixels=_positive_int("SYNTHDET_MAX_IMAGE_PIXELS", 25_000_000),
            inference_timeout_seconds=_positive_int("SYNTHDET_INFERENCE_TIMEOUT_SECONDS", 60),
            max_batch_images=min(_positive_int("SYNTHDET_MAX_BATCH_IMAGES", 4), 8),
            allowed_origins=origins,
        )
