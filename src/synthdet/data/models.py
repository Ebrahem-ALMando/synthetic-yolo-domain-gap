"""Shared records for real-dataset validation and manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    width_px: float
    height_px: float
    area_ratio: float
    size_group: str


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str
    message: str
    image_path: str | None = None
    label_path: str | None = None
    line_number: int | None = None
    before: str | None = None
    after: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageRecord:
    image_path: str
    label_path: str | None
    content_hash: str
    perceptual_hash: str
    width: int
    height: int
    aspect_ratio: float | None
    classes_present: list[str]
    object_count: int
    boxes: list[BoundingBox] = field(default_factory=list)
    inclusion_status: str = "included"
    exclusion_reasons: list[str] = field(default_factory=list)
    source_group_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["classes_present"] = ";".join(self.classes_present)
        data["exclusion_reasons"] = ";".join(self.exclusion_reasons)
        data["boxes"] = [asdict(box) for box in self.boxes]
        return data
