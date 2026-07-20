"""Public API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    project: str
    evaluation_status: str


class Detection(BaseModel):
    class_id: int
    class_name: str
    class_name_ar: str
    confidence: float
    bbox_xyxy_pixels: list[float] = Field(min_length=4, max_length=4)
    bbox_xyxy_normalized: list[float] = Field(min_length=4, max_length=4)


class InferenceResponse(BaseModel):
    model_id: str
    filename: str
    original_width: int
    original_height: int
    detections: list[Detection]
    detection_count: int
    preprocessing_duration_ms: float
    inference_duration_ms: float
    postprocessing_duration_ms: float
    total_duration_ms: float
    device: str
    annotated_image_mime: str | None = None
    annotated_image_base64: str | None = None
