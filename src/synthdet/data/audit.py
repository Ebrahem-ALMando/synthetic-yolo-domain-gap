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


def _save_plots(
    output_dir: Path,
    included: list[dict[str, str]],
    included_boxes: list[dict[str, str]],
    class_names: list[str],
    class_objects: Counter[str],
    images_per_class: Counter[str],
    size_groups: Counter[str],
) -> list[str]:
    """Render compact plots exclusively from inspected, included records."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    def save(name: str) -> None:
        plt.tight_layout()
        target = plot_dir / name
        plt.savefig(target, dpi=160, bbox_inches="tight")
        plt.close()
        created.append(target.relative_to(output_dir).as_posix())

    positions = list(range(len(class_names)))
    plt.figure(figsize=(9, 4.8))
    plt.bar(positions, [class_objects[name] for name in class_names], color="#2563eb")
    plt.xticks(positions, class_names, rotation=30, ha="right")
    plt.ylabel("Annotated objects")
    plt.title("Included object count by class")
    save("class_object_distribution.png")

    plt.figure(figsize=(9, 4.8))
    plt.bar(positions, [images_per_class[name] for name in class_names], color="#0f766e")
    plt.xticks(positions, class_names, rotation=30, ha="right")
    plt.ylabel("Included images")
    plt.title("Included images containing each class")
    save("images_per_class.png")

    object_counts = [int(record["object_count"]) for record in included]
    plt.figure(figsize=(7.5, 4.8))
    plt.hist(object_counts, bins=range(1, max(object_counts) + 2), color="#7c3aed")
    plt.xlabel("Valid objects per image")
    plt.ylabel("Images")
    plt.title("Objects per included image")
    save("objects_per_image.png")

    widths = [int(record["width"]) for record in included]
    heights = [int(record["height"]) for record in included]
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(widths, heights, alpha=0.25, s=18, color="#ea580c")
    plt.xlabel("Image width (px)")
    plt.ylabel("Image height (px)")
    plt.title("Included image resolutions")
    save("image_resolution.png")

    aspect_ratios = [float(record["aspect_ratio"]) for record in included]
    plt.figure(figsize=(7.5, 4.8))
    plt.hist(aspect_ratios, bins=30, color="#0891b2")
    plt.xlabel("Width / height")
    plt.ylabel("Images")
    plt.title("Included image aspect ratios")
    save("image_aspect_ratio.png")

    box_widths = [float(box["width_px"]) for box in included_boxes]
    box_heights = [float(box["height_px"]) for box in included_boxes]
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(box_widths, box_heights, alpha=0.12, s=10, color="#be123c")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Bounding-box width (px, log scale)")
    plt.ylabel("Bounding-box height (px, log scale)")
    plt.title("Included bounding-box dimensions")
    save("bounding_box_dimensions.png")

    area_ratios = [float(box["area_ratio"]) for box in included_boxes]
    plt.figure(figsize=(7.5, 4.8))
    plt.hist(area_ratios, bins=50, color="#4d7c0f")
    plt.xscale("log")
    plt.xlabel("Bounding-box area / image area (log scale)")
    plt.ylabel("Objects")
    plt.title("Included bounding-box area ratios")
    save("bounding_box_area_ratio.png")

    size_names = ["small", "medium", "large"]
    plt.figure(figsize=(6.5, 4.8))
    plt.bar(size_names, [size_groups[name] for name in size_names], color="#9333ea")
    plt.ylabel("Annotated objects")
    plt.title("Included object-size distribution")
    save("object_size_distribution.png")
    return created


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
        "discovered_label_files": issue_payload.get("discovered_label_files"),
        "matched_image_label_pairs": issue_payload.get("matched_image_label_pairs"),
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
    statistics["plot_files"] = _save_plots(
        output_dir,
        included,
        included_boxes,
        class_names,
        class_objects,
        images_per_class,
        size_groups,
    )
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
        f"- Discovered label files: {statistics['discovered_label_files']}",
        f"- Matched image-label pairs: {statistics['matched_image_label_pairs']}",
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
            "## Image and object statistics",
            "",
            f"- Objects per included image: minimum {statistics['objects_per_image']['minimum']}, "
            f"maximum {statistics['objects_per_image']['maximum']}, mean "
            f"{statistics['objects_per_image']['mean']:.4f}, median "
            f"{statistics['objects_per_image']['median']}.",
            f"- Image width range: {statistics['image_width']['minimum']} to "
            f"{statistics['image_width']['maximum']} pixels.",
            f"- Image height range: {statistics['image_height']['minimum']} to "
            f"{statistics['image_height']['maximum']} pixels.",
            f"- Aspect ratio range: {statistics['image_aspect_ratio']['minimum']:.4f} to "
            f"{statistics['image_aspect_ratio']['maximum']:.4f}; mean "
            f"{statistics['image_aspect_ratio']['mean']:.4f}, median "
            f"{statistics['image_aspect_ratio']['median']:.4f}.",
            f"- Bounding-box area-ratio range: "
            f"{statistics['bounding_box_area_ratio']['minimum']:.10f} to "
            f"{statistics['bounding_box_area_ratio']['maximum']:.10f}; mean "
            f"{statistics['bounding_box_area_ratio']['mean']:.6f}, median "
            f"{statistics['bounding_box_area_ratio']['median']:.6f}.",
            f"- Class-imbalance max/min nonzero object ratio: "
            f"{statistics['class_imbalance_max_to_min_nonzero']:.4f}.",
            "",
            "## Object-size rule",
            "",
            "Small objects have pixel area below 32^2, medium objects have pixel area from 32^2 "
            "up to but excluding 96^2, and large objects have pixel area at least 96^2. Areas are "
            "computed in each inspected image's pixel coordinate system.",
            "",
            f"Measured counts: small {statistics['object_size_counts']['small']}, medium "
            f"{statistics['object_size_counts']['medium']}, large "
            f"{statistics['object_size_counts']['large']}.",
            "",
            "## Validation issues",
            "",
        ]
    )
    if issue_counts:
        lines.extend(f"- `{code}`: {count}" for code, count in sorted(issue_counts.items()))
    else:
        lines.append("- No validation issues were recorded.")
    lines.extend(["", "## Generated plots", ""])
    lines.extend(f"- `{path}`" for path in statistics["plot_files"])
    (output_dir / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return statistics
