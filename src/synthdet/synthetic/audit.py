"""Data-derived statistics and visual review sheets for synthetic composites."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from synthdet.synthetic.contracts import SyntheticConfig, read_csv, write_csv
from synthdet.synthetic.geometry import yolo_to_xyxy


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(array.min()) if len(array) else 0.0,
        "mean": float(array.mean()) if len(array) else 0.0,
        "median": float(np.median(array)) if len(array) else 0.0,
        "maximum": float(array.max()) if len(array) else 0.0,
    }


def _bar_chart(values: dict[str, int | float], title: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    labels = list(values)
    heights = [values[label] for label in labels]
    axis.bar(labels, heights, color="#277da1")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _actual(path: str, logical_root: Path, physical_root: Path) -> Path:
    return physical_root / Path(path).relative_to(logical_root)


def _annotated_image(image: Image.Image, label_path: Path, class_names: list[str]) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        class_id = int(values[0])
        box = yolo_to_xyxy(*(float(value) for value in values[1:]), output.width, output.height)
        draw.rectangle(box, outline="red", width=max(2, output.width // 300))
        draw.text((box[0] + 2, box[1] + 2), class_names[class_id], fill="yellow")
    return output


def _quality_contact_sheets(
    config: SyntheticConfig,
    project_root: Path,
    physical_output_root: Path,
    logical_output_root: Path,
    bank_physical_output_root: Path,
    bank_manifest_dir: Path,
    manifest_dir: Path,
    report_dir: Path,
) -> list[str]:
    images = read_csv(manifest_dir / "synthetic_images.csv")
    sources = read_csv(manifest_dir / "synthetic_sources.csv")
    bank_by_id = {
        row["object_id"]: row
        for row in read_csv(bank_manifest_dir / "object_bank.csv")
        if row["inclusion_status"] == "included"
    }
    image_by_path = {row["synthetic_image_path"]: row for row in images}
    selected: list[dict[str, str]] = []
    if len(images) <= 16:
        for image_row in images:
            selected.append(
                next(
                    row
                    for row in sources
                    if row["synthetic_image_path"] == image_row["synthetic_image_path"]
                )
            )
    else:
        for class_id in range(len(config.class_names)):
            selected.append(
                next(row for row in sources if int(row["class_id"]) == class_id)
            )
        predicates = (
            lambda row: row["horizontal_flip"] == "true",
            lambda row: float(row["blur_radius"]) > 0,
            lambda row: float(row["noise_standard_deviation"]) > 0,
            lambda row: abs(float(row["rotation_degrees"])) >= 8,
        )
        for predicate in predicates:
            candidate = next((row for row in sources if predicate(row)), None)
            if candidate is not None and candidate not in selected:
                selected.append(candidate)
    outputs: list[str] = []
    logical_bank_root = Path(config.dataset["output_directory"])
    page_size = 4
    panel_width, panel_height, label_height = 230, 180, 58
    for offset in range(0, len(selected), page_size):
        page = selected[offset : offset + page_size]
        canvas = Image.new(
            "RGB", (4 * panel_width, len(page) * (panel_height + label_height)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, source_row in enumerate(page):
            synthetic = image_by_path[source_row["synthetic_image_path"]]
            with Image.open(project_root / synthetic["base_image_path"]) as base_source:
                base = ImageOps.contain(
                    base_source.convert("RGB"), (panel_width - 8, panel_height - 8)
                )
            bank = bank_by_id[source_row["object_id"]]
            crop_path = _actual(
                bank["crop_output_path"], logical_bank_root, bank_physical_output_root
            )
            mask_path = _actual(
                bank["mask_output_path"], logical_bank_root, bank_physical_output_root
            )
            with Image.open(crop_path) as crop_source:
                crop = ImageOps.contain(
                    crop_source.convert("RGB"), (panel_width - 8, panel_height - 8)
                )
            with Image.open(mask_path) as mask_source:
                mask = ImageOps.contain(
                    mask_source.convert("L"), (panel_width - 8, panel_height - 8)
                ).convert("RGB")
            final_path = _actual(
                synthetic["synthetic_image_path"], logical_output_root, physical_output_root
            )
            final_label = _actual(
                synthetic["synthetic_label_path"], logical_output_root, physical_output_root
            )
            with Image.open(final_path) as final_source:
                final = ImageOps.contain(
                    _annotated_image(final_source, final_label, config.class_names),
                    (panel_width - 8, panel_height - 8),
                )
            panels = (base, crop, mask, final)
            y = row_index * (panel_height + label_height)
            for panel_index, panel in enumerate(panels):
                x = panel_index * panel_width + (panel_width - panel.width) // 2
                canvas.paste(panel, (x, y + (panel_height - panel.height) // 2))
            draw.text(
                (5, y + panel_height + 2),
                f"{source_row['class_name']} | {source_row['object_id']} | "
                "base / crop / mask / final+boxes",
                fill="black",
            )
            draw.text(
                (5, y + panel_height + 22),
                f"scale={float(source_row['scale']):.2f} "
                f"rot={float(source_row['rotation_degrees']):.1f} "
                f"flip={source_row['horizontal_flip']} blur={float(source_row['blur_radius']):.2f} "
                f"noise={float(source_row['noise_standard_deviation']):.2f}",
                fill="#555555",
            )
            draw.text(
                (5, y + panel_height + 40),
                f"IoU={float(source_row['maximum_iou']):.3f} "
                f"retries={source_row['placement_retries']}",
                fill="#555555",
            )
        target = (
            report_dir
            / "visual_review"
            / f"synthetic-quality-page-{offset // page_size + 1:02d}.jpg"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, quality=92)
        outputs.append(target.as_posix())
    return outputs


def generate_synthetic_audit(
    config: SyntheticConfig,
    project_root: Path,
    physical_output_root: Path,
    logical_output_root: Path,
    bank_physical_output_root: Path,
    bank_manifest_dir: Path,
    manifest_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    images = read_csv(manifest_dir / "synthetic_images.csv")
    sources = read_csv(manifest_dir / "synthetic_sources.csv")
    bank = read_csv(bank_manifest_dir / "object_bank.csv")
    excluded = read_csv(bank_manifest_dir / "excluded_objects.csv")
    failed = read_csv(manifest_dir / "failed_generation_attempts.csv")
    metadata = json.loads((manifest_dir / "generation_metadata.json").read_text(encoding="utf-8"))
    total_class_counts: Counter[str] = Counter()
    images_per_class: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    for row in images:
        label_path = _actual(row["synthetic_label_path"], logical_output_root, physical_output_root)
        classes_in_image: set[str] = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            class_name = config.class_names[int(values[0])]
            total_class_counts[class_name] += 1
            classes_in_image.add(class_name)
            width_px = float(values[3]) * int(row["image_width"])
            height_px = float(values[4]) * int(row["image_height"])
            area = width_px * height_px
            size_counts["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
        images_per_class.update(classes_in_image)
        resolution_counts[f"{row['image_width']}x{row['image_height']}"] += 1
    pasted_class_counts = Counter(row["class_name"] for row in sources)
    target = metadata["target_pasted_class_proportions"]
    actual = metadata["actual_pasted_class_proportions"]
    deviations = {name: actual[name] - target[name] for name in config.class_names}
    method_counts = Counter(
        row["extraction_method"] for row in bank if row["inclusion_status"] == "included"
    )
    transformation_usage = {
        "horizontal_flip": sum(row["horizontal_flip"] == "true" for row in sources),
        "rotation_nonzero": sum(abs(float(row["rotation_degrees"])) > 1e-9 for row in sources),
        "brightness_jitter": sum(abs(float(row["brightness"]) - 1) > 1e-9 for row in sources),
        "contrast_jitter": sum(abs(float(row["contrast"]) - 1) > 1e-9 for row in sources),
        "saturation_jitter": sum(abs(float(row["saturation"]) - 1) > 1e-9 for row in sources),
        "blur": sum(float(row["blur_radius"]) > 0 for row in sources),
        "noise": sum(float(row["noise_standard_deviation"]) > 0 for row in sources),
        "jpeg_compression": len(images),
        "alpha_feather": len(sources),
    }
    output_hashes = [row["output_image_hash"] for row in images]
    real_hashes = {
        row["content_hash"]
        for split in ("train", "val", "test")
        for row in read_csv(
            project_root / Path(config.dataset["active_split_directory"]) / f"real_{split}.csv"
        )
    }
    contact_sheets = _quality_contact_sheets(
        config,
        project_root,
        physical_output_root,
        logical_output_root,
        bank_physical_output_root,
        bank_manifest_dir,
        manifest_dir,
        report_dir,
    )
    statistics = {
        "synthetic_image_count": len(images),
        "pasted_object_count": len(sources),
        "total_resulting_object_count": sum(int(row["total_object_count"]) for row in images),
        "retained_base_object_count": sum(int(row["retained_base_object_count"]) for row in images),
        "total_class_counts": dict(sorted(total_class_counts.items())),
        "pasted_class_counts": dict(sorted(pasted_class_counts.items())),
        "images_per_class": dict(sorted(images_per_class.items())),
        "pasted_objects_per_image": _summary([int(row["pasted_object_count"]) for row in images]),
        "retained_base_objects_per_image": _summary(
            [int(row["retained_base_object_count"]) for row in images]
        ),
        "object_size_distribution": dict(sorted(size_counts.items())),
        "transformation_usage": transformation_usage,
        "placement_retries": _summary([int(row["placement_retries"]) for row in sources]),
        "total_rejected_placement_attempts": metadata["rejected_placement_attempt_count"],
        "failed_sample_attempts": len(failed),
        "generation_failure_reasons": dict(
            sorted(Counter(row["reason"] for row in failed).items())
        ),
        "object_bank_exclusion_reasons": dict(
            sorted(Counter(row["exclusion_reason"] for row in excluded).items())
        ),
        "extraction_method_usage": dict(sorted(method_counts.items())),
        "mask_coverage": _summary(
            [float(row["mask_coverage"]) for row in bank if row["inclusion_status"] == "included"]
        ),
        "maximum_overlap_iou": _summary([float(row["maximum_iou"]) for row in sources]),
        "maximum_existing_occlusion": _summary(
            [float(row["maximum_existing_occlusion"]) for row in sources]
        ),
        "output_resolution_distribution": dict(sorted(resolution_counts.items())),
        "exact_duplicate_synthetic_outputs": len(output_hashes) - len(set(output_hashes)),
        "exact_real_image_collisions": len(real_hashes & set(output_hashes)),
        "target_pasted_class_proportions": target,
        "actual_pasted_class_proportions": actual,
        "pasted_class_proportion_deviation": deviations,
        "visual_review_contact_sheets": contact_sheets,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "synthetic_quality_statistics.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(
        report_dir / "class_distribution.csv",
        [
            "class_name",
            "total_objects",
            "pasted_objects",
            "images",
            "target",
            "actual",
            "deviation",
        ],
        [
            {
                "class_name": name,
                "total_objects": total_class_counts[name],
                "pasted_objects": pasted_class_counts[name],
                "images": images_per_class[name],
                "target": f"{target[name]:.10f}",
                "actual": f"{actual[name]:.10f}",
                "deviation": f"{deviations[name]:.10f}",
            }
            for name in config.class_names
        ],
    )
    _bar_chart(
        dict(total_class_counts),
        "Synthetic total object distribution",
        report_dir / "total_class_distribution.png",
    )
    _bar_chart(
        dict(pasted_class_counts),
        "Pasted object distribution",
        report_dir / "pasted_class_distribution.png",
    )
    _bar_chart(
        dict(size_counts),
        "Synthetic object-size distribution",
        report_dir / "object_size_distribution.png",
    )
    _bar_chart(
        transformation_usage, "Transformation usage", report_dir / "transformation_usage.png"
    )
    _bar_chart(
        dict(method_counts), "Object extraction methods", report_dir / "extraction_methods.png"
    )
    return statistics
