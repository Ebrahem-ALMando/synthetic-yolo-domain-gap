"""Data-derived tables and figures for a frozen real-data split."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _bar_figure(
    output: Path,
    labels: list[str],
    series: dict[str, list[int]],
    title: str,
    ylabel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    width = 0.8 / max(len(series), 1)
    positions = list(range(len(labels)))
    for index, (name, values) in enumerate(series.items()):
        offsets = [position - 0.4 + width / 2 + index * width for position in positions]
        axis.bar(offsets, values, width=width, label=name)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if len(series) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def generate_split_audit(
    manifest_dir: Path,
    bounding_boxes_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate split statistics, small tables, and six real-data figures."""

    split_names = ("train", "val", "test")
    manifests = {
        split: _read_csv(manifest_dir / f"real_{split}.csv") for split in split_names
    }
    path_to_split = {
        row["image_path"]: split for split, rows in manifests.items() for row in rows
    }
    boxes: list[dict[str, str]] = []
    for row in _read_csv(bounding_boxes_path):
        image_path = row["image_path"]
        matches = [
            path
            for path in path_to_split
            if path == image_path or path.endswith(f"/{image_path}")
        ]
        if len(matches) == 1:
            box = dict(row)
            box["_split"] = path_to_split[matches[0]]
            boxes.append(box)
    class_names = sorted(
        {
            class_name
            for rows in manifests.values()
            for row in rows
            for class_name in row["classes_present"].split(";")
            if class_name
        }
        | {row["class_name"] for row in boxes}
    )
    size_names = ("small", "medium", "large")
    total_images = sum(len(rows) for rows in manifests.values())

    image_class_counts: dict[str, Counter[str]] = {}
    object_class_counts: dict[str, Counter[str]] = {}
    object_size_counts: dict[str, Counter[str]] = {}
    source_group_counts: dict[str, int] = {}
    object_counts: dict[str, int] = {}
    for split, rows in manifests.items():
        image_classes: Counter[str] = Counter()
        for row in rows:
            image_classes.update(value for value in row["classes_present"].split(";") if value)
        split_boxes = [row for row in boxes if row["_split"] == split]
        image_class_counts[split] = image_classes
        object_class_counts[split] = Counter(row["class_name"] for row in split_boxes)
        object_size_counts[split] = Counter(row["size_group"] for row in split_boxes)
        source_group_counts[split] = len({row["source_group_id"] for row in rows})
        object_counts[split] = len(split_boxes)

    statistics: dict[str, Any] = {
        "total_images": total_images,
        "total_objects": len(boxes),
        "image_counts": {split: len(manifests[split]) for split in split_names},
        "image_percentages": {
            split: len(manifests[split]) / total_images for split in split_names
        },
        "source_group_counts": source_group_counts,
        "object_counts": object_counts,
        "image_count_per_class": {
            split: {name: image_class_counts[split][name] for name in class_names}
            for split in split_names
        },
        "object_count_per_class": {
            split: {name: object_class_counts[split][name] for name in class_names}
            for split in split_names
        },
        "object_size_counts": {
            split: {name: object_size_counts[split][name] for name in size_names}
            for split in split_names
        },
        "class_coverage_limitations": {
            split: [name for name in class_names if image_class_counts[split][name] == 0]
            for split in split_names
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_dir / "split_summary.csv",
        ["split", "image_count", "image_percentage", "source_group_count", "object_count"],
        [
            {
                "split": split,
                "image_count": len(manifests[split]),
                "image_percentage": statistics["image_percentages"][split],
                "source_group_count": source_group_counts[split],
                "object_count": object_counts[split],
            }
            for split in split_names
        ],
    )
    _write_csv(
        output_dir / "per_class_split_statistics.csv",
        ["class_name", "split", "image_count", "object_count"],
        [
            {
                "class_name": name,
                "split": split,
                "image_count": image_class_counts[split][name],
                "object_count": object_class_counts[split][name],
            }
            for name in class_names
            for split in split_names
        ],
    )
    _write_csv(
        output_dir / "object_size_split_statistics.csv",
        ["size_group", *split_names],
        [
            {
                "size_group": name,
                **{split: object_size_counts[split][name] for split in split_names},
            }
            for name in size_names
        ],
    )

    _bar_figure(
        output_dir / "split_image_counts.png",
        list(split_names),
        {"images": [len(manifests[split]) for split in split_names]},
        "Real image count by split",
        "Images",
    )
    _bar_figure(
        output_dir / "per_class_image_distribution.png",
        class_names,
        {
            split: [image_class_counts[split][name] for name in class_names]
            for split in split_names
        },
        "Per-class image distribution by split",
        "Images containing class",
    )
    _bar_figure(
        output_dir / "per_class_object_distribution.png",
        class_names,
        {
            split: [object_class_counts[split][name] for name in class_names]
            for split in split_names
        },
        "Per-class object distribution by split",
        "Objects",
    )
    _bar_figure(
        output_dir / "object_size_distribution.png",
        list(size_names),
        {
            split: [object_size_counts[split][name] for name in size_names]
            for split in split_names
        },
        "Object-size distribution by split",
        "Objects",
    )
    _bar_figure(
        output_dir / "total_object_distribution.png",
        list(split_names),
        {"objects": [object_counts[split] for split in split_names]},
        "Total object count by split",
        "Objects",
    )
    _bar_figure(
        output_dir / "source_group_distribution.png",
        list(split_names),
        {"source groups": [source_group_counts[split] for split in split_names]},
        "Stable source-group count by split",
        "Source groups",
    )

    lines = [
        "# Aquarium Immutable Split Audit",
        "",
        "| Split | Images | Percentage | Source groups | Objects |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in split_names:
        lines.append(
            f"| {split} | {len(manifests[split])} | "
            f"{statistics['image_percentages'][split]:.2%} | "
            f"{source_group_counts[split]} | {object_counts[split]} |"
        )
    lines.extend(
        [
            "",
            "Class coverage limitations: "
            + "; ".join(
                f"{split}={','.join(statistics['class_coverage_limitations'][split]) or 'none'}"
                for split in split_names
            )
            + ".",
            "",
            "All values were derived from the frozen manifests and audited bounding-box rows.",
        ]
    )
    (output_dir / "split_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return statistics
