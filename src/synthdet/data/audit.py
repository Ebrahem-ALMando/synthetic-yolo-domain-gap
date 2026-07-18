"""Statistics and human-readable summaries derived only from inspected files."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required validation output not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_dataset_audit(validation_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Generate factual statistics from validation CSV/JSON outputs."""

    records = _read_csv(validation_dir / "image_records.csv")
    boxes = _read_csv(validation_dir / "bounding_boxes.csv")
    issue_path = validation_dir / "validation_issues.json"
    if not issue_path.is_file():
        raise FileNotFoundError(f"Required validation output not found: {issue_path}")
    issue_payload = json.loads(issue_path.read_text(encoding="utf-8"))
    class_names = issue_payload["class_names"]
    included = [record for record in records if record["inclusion_status"] == "included"]
    included_paths = {record["image_path"] for record in included}
    included_boxes = [box for box in boxes if box["image_path"] in included_paths]

    class_objects = Counter(box["class_name"] for box in included_boxes)
    images_per_class: Counter[str] = Counter()
    for record in included:
        for class_name in filter(None, record["classes_present"].split(";")):
            images_per_class[class_name] += 1
    object_counts = [int(record["object_count"]) for record in included]
    widths = [int(record["width"]) for record in included]
    heights = [int(record["height"]) for record in included]
    aspect_ratios = [float(record["aspect_ratio"]) for record in included]
    area_ratios = [float(box["area_ratio"]) for box in included_boxes]
    size_groups = Counter(box["size_group"] for box in included_boxes)
    issue_counts = Counter(issue["code"] for issue in issue_payload["issues"])
    nonzero_class_counts = [class_objects[name] for name in class_names if class_objects[name] > 0]
    imbalance_ratio = (
        max(nonzero_class_counts) / min(nonzero_class_counts) if nonzero_class_counts else None
    )

    statistics: dict[str, Any] = {
        "inspected_images": len(records),
        "included_images": len(included),
        "excluded_images": len(records) - len(included),
        "images_without_valid_objects": sum(int(record["object_count"]) == 0 for record in records),
        "class_names": class_names,
        "per_class_object_counts": {name: class_objects[name] for name in class_names},
        "images_per_class": {name: images_per_class[name] for name in class_names},
        "class_imbalance_max_to_min_nonzero": imbalance_ratio,
        "objects_per_image": {
            "minimum": min(object_counts) if object_counts else None,
            "maximum": max(object_counts) if object_counts else None,
            "mean": mean(object_counts) if object_counts else None,
            "median": median(object_counts) if object_counts else None,
        },
        "image_width": {
            "minimum": min(widths) if widths else None,
            "maximum": max(widths) if widths else None,
        },
        "image_height": {
            "minimum": min(heights) if heights else None,
            "maximum": max(heights) if heights else None,
        },
        "image_aspect_ratio": {
            "minimum": min(aspect_ratios) if aspect_ratios else None,
            "maximum": max(aspect_ratios) if aspect_ratios else None,
            "mean": mean(aspect_ratios) if aspect_ratios else None,
            "median": median(aspect_ratios) if aspect_ratios else None,
        },
        "bounding_box_area_ratio": {
            "minimum": min(area_ratios) if area_ratios else None,
            "maximum": max(area_ratios) if area_ratios else None,
            "mean": mean(area_ratios) if area_ratios else None,
            "median": median(area_ratios) if area_ratios else None,
        },
        "object_size_counts": {
            size: size_groups[size] for size in ("small", "medium", "large")
        },
        "issue_counts": dict(sorted(issue_counts.items())),
        "object_size_rule": {
            "small": "pixel area < 32^2",
            "medium": "32^2 <= pixel area < 96^2",
            "large": "pixel area >= 96^2",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Aquarium Dataset Audit",
        "",
        "This report was generated from inspected local files. No value was entered manually.",
        "",
        "## Inclusion",
        "",
        f"- Inspected images: {statistics['inspected_images']}",
        f"- Included images: {statistics['included_images']}",
        f"- Excluded images: {statistics['excluded_images']}",
        f"- Images without valid objects: {statistics['images_without_valid_objects']}",
        "",
        "## Class distribution",
        "",
        "| Class | Objects | Images |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {name} | {class_objects[name]} | {images_per_class[name]} |" for name in class_names
    )
    lines.extend(
        [
            "",
            "## Object-size rule",
            "",
            "Small objects have pixel area below 32^2, medium objects have pixel area from 32^2 "
            "up to but excluding 96^2, and large objects have pixel area at least 96^2. Areas are "
            "computed in each inspected image's pixel coordinate system.",
            "",
            "## Validation issues",
            "",
        ]
    )
    if issue_counts:
        lines.extend(f"- `{code}`: {count}" for code, count in sorted(issue_counts.items()))
    else:
        lines.append("- No validation issues were recorded.")
    (output_dir / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return statistics
