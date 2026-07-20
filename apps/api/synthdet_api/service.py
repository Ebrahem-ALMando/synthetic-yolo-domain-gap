"""Secure lazy YOLO inference service."""

from __future__ import annotations

import base64
import hashlib
import io
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from synthdet_api.config import Settings
from synthdet_api.repository import CLASS_NAMES_AR, ProjectRepository

MIME_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


class APIError(Exception):
    def __init__(
        self, status_code: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


class ModelService:
    def __init__(self, settings: Settings, repository: ProjectRepository) -> None:
        self.settings = settings
        self.repository = repository
        self._lock = threading.Lock()
        self._model_id: str | None = None
        self._model: YOLO | None = None

    def device(self) -> str:
        policy = self.settings.device_policy
        if policy == "cuda" and not torch.cuda.is_available():
            raise APIError(503, "cuda_unavailable", "CUDA was requested but is unavailable")
        if policy == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return "cuda:0" if policy == "cuda" else "cpu"

    def _load(self, model_id: str) -> YOLO:
        record = self.repository.model_record(model_id)
        if not record["available"]:
            raise APIError(
                503,
                "model_unavailable",
                "The requested checkpoint is unavailable or failed its SHA-256 check",
                {"model_id": model_id, "reason": record["availability_reason"]},
            )
        with self._lock:
            if self._model_id != model_id or self._model is None:
                self._model = YOLO(str(self.repository.checkpoint_path(model_id)))
                self._model_id = model_id
            return self._model

    def validate_upload(self, data: bytes, filename: str, content_type: str) -> Image.Image:
        safe_name = Path(filename or "upload").name
        if safe_name != filename or safe_name in {"", ".", ".."}:
            raise APIError(400, "invalid_filename", "The upload filename is invalid")
        if content_type not in MIME_FORMATS:
            raise APIError(415, "unsupported_media_type", "Only JPEG, PNG, and WebP are accepted")
        if not data:
            raise APIError(400, "empty_upload", "The uploaded file is empty")
        if len(data) > self.settings.max_upload_bytes:
            raise APIError(
                413, "upload_too_large", "The uploaded file exceeds the configured limit"
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest in self.repository.protected_hashes:
            raise APIError(
                403,
                "protected_test_image",
                "This image belongs to the sealed real test set and cannot be reused interactively",
            )
        try:
            with Image.open(io.BytesIO(data)) as probe:
                detected_format = probe.format
                width, height = probe.size
                probe.verify()
            if detected_format != MIME_FORMATS[content_type]:
                raise APIError(
                    415, "mime_mismatch", "MIME type does not match decoded image format"
                )
            if width * height > self.settings.max_image_pixels:
                raise APIError(
                    413, "image_dimensions_too_large", "Decoded image dimensions are too large"
                )
            with Image.open(io.BytesIO(data)) as decoded:
                return decoded.convert("RGB")
        except APIError:
            raise
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as error:
            raise APIError(
                400, "invalid_image", "The upload is not a valid decodable image"
            ) from error

    def infer(
        self,
        *,
        model_id: str,
        image: Image.Image,
        filename: str,
        confidence: float,
        iou: float,
        max_detections: int,
        annotate: bool,
    ) -> dict[str, Any]:
        model = self._load(model_id)
        device = self.device()
        width, height = image.size
        started = time.perf_counter()
        results = model.predict(
            source=np.asarray(image),
            imgsz=640,
            conf=confidence,
            iou=iou,
            max_det=max_detections,
            device=device,
            verbose=False,
        )
        if len(results) != 1:
            raise APIError(500, "inference_result_error", "Inference returned an unexpected batch")
        result = results[0]
        detections = []
        for xyxy, score, class_id_value in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
            strict=True,
        ):
            class_id = int(class_id_value)
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.repository.contract["class_names"][class_id],
                    "class_name_ar": CLASS_NAMES_AR[class_id],
                    "confidence": float(score),
                    "bbox_xyxy_pixels": [x1, y1, x2, y2],
                    "bbox_xyxy_normalized": [x1 / width, y1 / height, x2 / width, y2 / height],
                }
            )
        annotated_mime = None
        annotated_base64 = None
        if annotate:
            plotted_bgr = result.plot()
            plotted_rgb = Image.fromarray(plotted_bgr[:, :, ::-1])
            buffer = io.BytesIO()
            plotted_rgb.save(buffer, format="PNG")
            annotated_mime = "image/png"
            annotated_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        speed = result.speed
        return {
            "model_id": model_id,
            "filename": Path(filename).name,
            "original_width": width,
            "original_height": height,
            "detections": detections,
            "detection_count": len(detections),
            "preprocessing_duration_ms": float(speed.get("preprocess", 0.0)),
            "inference_duration_ms": float(speed.get("inference", 0.0)),
            "postprocessing_duration_ms": float(speed.get("postprocess", 0.0)),
            "total_duration_ms": (time.perf_counter() - started) * 1000,
            "device": device,
            "annotated_image_mime": annotated_mime,
            "annotated_image_base64": annotated_base64,
        }
