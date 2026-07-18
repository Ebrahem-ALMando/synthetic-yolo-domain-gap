"""Strict validation and non-destructive normalization for YOLO detection data."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from synthdet.data.hashing import difference_hash, sha256_file
from synthdet.data.models import AuditIssue, BoundingBox, ImageRecord

SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SMALL_OBJECT_MAX_AREA_PX = 32**2
MEDIUM_OBJECT_MAX_AREA_PX = 96**2


def load_class_names(dataset_root: Path) -> list[str]:
    """Read the stable class order from the acquired export metadata."""

    candidates = sorted(dataset_root.rglob("data.yaml")) + sorted(dataset_root.rglob("data.yml"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one data.yaml/data.yml under {dataset_root}, found {len(candidates)}"
        )
    raw: Any = yaml.safe_load(candidates[0].read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "names" not in raw:
        raise ValueError(f"Dataset metadata has no 'names' value: {candidates[0]}")
    names = raw["names"]
    if isinstance(names, list):
        result = [str(name).strip() for name in names]
    elif isinstance(names, dict):
        try:
            ordered_ids = sorted(int(class_id) for class_id in names)
            if ordered_ids != list(range(len(ordered_ids))):
                raise ValueError("class IDs in data.yaml must be contiguous and zero-based")
            result = [
                str(names.get(class_id, names.get(str(class_id)))).strip()
                for class_id in ordered_ids
            ]
        except (TypeError, ValueError) as error:
            raise ValueError("data.yaml names mapping must use integer class IDs") from error
    else:
        raise ValueError("data.yaml names must be a list or class-ID mapping")
    if not result or any(not name for name in result) or len(result) != len(set(result)):
        raise ValueError("data.yaml class names must be non-empty and unique")
    return result


def _label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if image_indexes:
        parts[image_indexes[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _size_group(width_px: float, height_px: float) -> str:
    area = width_px * height_px
    if area < SMALL_OBJECT_MAX_AREA_PX:
        return "small"
    if area < MEDIUM_OBJECT_MAX_AREA_PX:
        return "medium"
    return "large"


def _parse_label(
    label_path: Path,
    label_relative: str,
    image_relative: str,
    width: int,
    height: int,
    class_names: list[str],
) -> tuple[list[BoundingBox], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    boxes: list[BoundingBox] = []
    lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    if not lines or all(not line.strip() for line in lines):
        issues.append(
            AuditIssue(
                "empty_label",
                "error",
                "Label file contains no annotation lines",
                image_relative,
                label_relative,
            )
        )
        return boxes, issues

    for line_number, raw_line in enumerate(lines, start=1):
        tokens = raw_line.split()
        if len(tokens) != 5:
            issues.append(
                AuditIssue(
                    "unsupported_annotation_line",
                    "error",
                    "YOLO detection rows must contain exactly five values",
                    image_relative,
                    label_relative,
                    line_number,
                    before=raw_line,
                )
            )
            continue
        try:
            class_value = float(tokens[0])
            coordinates = [float(value) for value in tokens[1:]]
        except ValueError:
            issues.append(
                AuditIssue(
                    "non_numeric_annotation",
                    "error",
                    "Annotation values must be numeric",
                    image_relative,
                    label_relative,
                    line_number,
                    before=raw_line,
                )
            )
            continue
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < len(class_names):
            issues.append(
                AuditIssue(
                    "unknown_class",
                    "error",
                    f"Class ID {tokens[0]} is not valid for {len(class_names)} classes",
                    image_relative,
                    label_relative,
                    line_number,
                    before=raw_line,
                )
            )
            continue
        x_center, y_center, box_width, box_height = coordinates
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            issues.append(
                AuditIssue(
                    "non_normalized_box",
                    "error",
                    "YOLO box coordinates must all be within [0, 1]",
                    image_relative,
                    label_relative,
                    line_number,
                    before=raw_line,
                )
            )
            continue
        if box_width <= 0.0 or box_height <= 0.0:
            issues.append(
                AuditIssue(
                    "non_positive_box",
                    "error",
                    "Bounding-box width and height must be positive",
                    image_relative,
                    label_relative,
                    line_number,
                    before=raw_line,
                )
            )
            continue
        if (
            x_center - box_width / 2 < 0.0
            or x_center + box_width / 2 > 1.0
            or y_center - box_height / 2 < 0.0
            or y_center + box_height / 2 > 1.0
        ):
            issues.append(
                AuditIssue(
                    "box_outside_image",
                    "error",
                    "Bounding box extends beyond image boundaries",
                    image_relative,
                    label_relative,
                    line_number,
                    before=raw_line,
                )
            )
            continue
        width_px = box_width * width
        height_px = box_height * height
        boxes.append(
            BoundingBox(
                class_id=class_id,
                class_name=class_names[class_id],
                x_center=x_center,
                y_center=y_center,
                width=box_width,
                height=box_height,
                width_px=width_px,
                height_px=height_px,
                area_ratio=box_width * box_height,
                size_group=_size_group(width_px, height_px),
            )
        )
    return boxes, issues


def validate_yolo_dataset(
    dataset_root: Path,
) -> tuple[list[str], list[ImageRecord], list[AuditIssue]]:
    """Inspect a YOLO export without changing any source file."""

    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    class_names = load_class_names(dataset_root)
    issues: list[AuditIssue] = []
    images: list[Path] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(path)
        elif any(part.lower() == "images" for part in path.parts):
            issues.append(
                AuditIssue(
                    "unsupported_image_extension",
                    "error",
                    f"Unsupported image extension: {path.suffix or '<none>'}",
                    path.relative_to(dataset_root).as_posix(),
                )
            )
    if not images:
        raise ValueError(f"No supported images found under {dataset_root}")

    records: list[ImageRecord] = []
    stem_paths: dict[str, list[str]] = defaultdict(list)
    expected_labels = {_label_for_image(image_path).resolve() for image_path in images}
    for label_path in sorted(dataset_root.rglob("*.txt")):
        if (
            any(part.lower() == "labels" for part in label_path.parts)
            and label_path.resolve() not in expected_labels
        ):
            issues.append(
                AuditIssue(
                    "orphan_label",
                    "error",
                    "Label file has no matching supported image",
                    label_path=label_path.relative_to(dataset_root).as_posix(),
                )
            )
    for image_path in images:
        relative_image = image_path.relative_to(dataset_root).as_posix()
        stem_paths[image_path.stem.casefold()].append(relative_image)
        label_path = _label_for_image(image_path)
        relative_label = (
            label_path.relative_to(dataset_root).as_posix()
            if label_path.is_relative_to(dataset_root)
            else None
        )
        image_issues: list[AuditIssue] = []
        boxes: list[BoundingBox] = []
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
                perceptual_hash = difference_hash(image)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            corrupt_issue = AuditIssue(
                "corrupt_image", "error", str(error), relative_image, relative_label
            )
            issues.append(corrupt_issue)
            records.append(
                ImageRecord(
                    image_path=relative_image,
                    label_path=relative_label,
                    content_hash=sha256_file(image_path),
                    perceptual_hash="",
                    width=0,
                    height=0,
                    aspect_ratio=None,
                    classes_present=[],
                    object_count=0,
                    inclusion_status="excluded",
                    exclusion_reasons=[corrupt_issue.code],
                )
            )
            continue

        if not label_path.is_file():
            image_issues.append(
                AuditIssue(
                    "missing_label",
                    "error",
                    "No matching YOLO label file",
                    relative_image,
                    relative_label,
                )
            )
        else:
            parsed_boxes, parse_issues = _parse_label(
                label_path,
                relative_label or label_path.name,
                relative_image,
                width,
                height,
                class_names,
            )
            boxes.extend(parsed_boxes)
            image_issues.extend(parse_issues)
        if not boxes:
            image_issues.append(
                AuditIssue(
                    "no_valid_objects",
                    "error",
                    "Image has no valid object annotations",
                    relative_image,
                    relative_label,
                )
            )
        exclusion_reasons = sorted(
            {issue.code for issue in image_issues if issue.severity == "error"}
        )
        records.append(
            ImageRecord(
                image_path=relative_image,
                label_path=relative_label,
                content_hash=sha256_file(image_path),
                perceptual_hash=perceptual_hash,
                width=width,
                height=height,
                aspect_ratio=width / height,
                classes_present=sorted({box.class_name for box in boxes}),
                object_count=len(boxes),
                boxes=boxes,
                inclusion_status="excluded" if exclusion_reasons else "included",
                exclusion_reasons=exclusion_reasons,
            )
        )
        issues.extend(image_issues)

    duplicate_stems = {stem: paths for stem, paths in stem_paths.items() if len(paths) > 1}
    for paths in duplicate_stems.values():
        for image_path in paths:
            issues.append(
                AuditIssue(
                    "duplicate_file_stem",
                    "error",
                    "File stem is not unique across the export",
                    image_path,
                )
            )
    if duplicate_stems:
        duplicate_paths = {path for paths in duplicate_stems.values() for path in paths}
        for record in records:
            if record.image_path in duplicate_paths:
                record.inclusion_status = "excluded"
                record.exclusion_reasons = sorted(
                    {*record.exclusion_reasons, "duplicate_file_stem"}
                )
    return class_names, records, issues


def write_validation_outputs(
    output_dir: Path,
    class_names: list[str],
    records: list[ImageRecord],
    issues: list[AuditIssue],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": class_names,
        "image_count": len(records),
        "included_count": sum(record.inclusion_status == "included" for record in records),
        "excluded_count": sum(record.inclusion_status == "excluded" for record in records),
        "issues": [issue.to_dict() for issue in issues],
    }
    (output_dir / "validation_issues.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    field_names = [field.name for field in fields(ImageRecord) if field.name != "boxes"]
    with (output_dir / "image_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            row = record.to_dict()
            row.pop("boxes")
            writer.writerow(row)
    with (output_dir / "bounding_boxes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        box_fields = ["image_path", *[field.name for field in fields(BoundingBox)]]
        writer = csv.DictWriter(handle, fieldnames=box_fields)
        writer.writeheader()
        for record in records:
            for box in record.boxes:
                writer.writerow({"image_path": record.image_path, **box.__dict__})


def normalize_included_records(
    dataset_root: Path, working_root: Path, records: list[ImageRecord]
) -> int:
    """Copy validated records to a working tree; source files remain untouched."""

    if working_root.exists() and any(working_root.iterdir()):
        raise FileExistsError(f"Working directory must be empty: {working_root}")
    copied = 0
    for record in records:
        if record.inclusion_status != "included" or record.label_path is None:
            continue
        image_source = dataset_root / record.image_path
        label_source = dataset_root / record.label_path
        image_target = working_root / record.image_path
        label_target = working_root / record.label_path
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_source, image_target)
        shutil.copy2(label_source, label_target)
        copied += 1
    return copied


def issue_counts(issues: list[AuditIssue]) -> Counter[str]:
    return Counter(issue.code for issue in issues)
