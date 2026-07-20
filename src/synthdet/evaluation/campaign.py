"""Locked Sprint 5 protected-test campaign execution and deterministic reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version

from synthdet.evaluation.contract import REGIMES, load_contract, validate_contract
from synthdet.synthetic.contracts import sha256_file
from synthdet.training.intake import CLASS_NAMES

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SIZE_NAMES = ("small", "medium", "large")
REAL_PERCENTAGES = {
    "synthetic_only": 0.0,
    "real_25": 25.058548,
    "real_50": 50.117096,
    "real_75": 74.941452,
    "real_only": 100.0,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_repository_path(root: Path, value: str, label: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"Unsafe {label} path: {value}")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository: {value}") from error
    return path


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_labels(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = raw.split()
        if len(values) != 5:
            raise ValueError(f"Malformed label row at {path}:{line_number}")
        class_id = int(values[0])
        coords = [float(value) for value in values[1:]]
        if class_id not in range(len(CLASS_NAMES)) or any(
            not math.isfinite(value) or value < 0 or value > 1 for value in coords
        ):
            raise ValueError(f"Invalid class/coordinate at {path}:{line_number}")
        x_center, y_center, box_width, box_height = coords
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Non-positive bounding box at {path}:{line_number}")
        x = (x_center - box_width / 2) * width
        y = (y_center - box_height / 2) * height
        objects.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "bbox_xywh_normalized": coords,
                "bbox_xywh_pixels": [x, y, box_width * width, box_height * height],
                "area_pixels": box_width * width * box_height * height,
            }
        )
    return objects


def validate_manifest_files(
    root: Path,
    manifest_path: Path,
    *,
    expected_split: str,
    expected_count: int,
    verify_pixels_and_labels: bool,
) -> list[dict[str, Any]]:
    """Validate one immutable split; pixel/label access is explicit at the call site."""

    rows = _read_manifest(manifest_path)
    if len(rows) != expected_count:
        raise ValueError(
            f"{expected_split} manifest count is {len(rows)}, expected {expected_count}"
        )
    required = {
        "image_path",
        "label_path",
        "content_hash",
        "source_group_id",
        "image_width",
        "image_height",
        "object_count",
        "split",
        "inclusion_status",
    }
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{expected_split} manifest is missing required fields")
    image_paths: set[str] = set()
    label_paths: set[str] = set()
    hashes: set[str] = set()
    basenames: set[str] = set()
    validated: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != expected_split or row["inclusion_status"] != "included":
            raise ValueError(f"Unexpected status/split in {expected_split} manifest")
        if (
            row["image_path"] in image_paths
            or row["label_path"] in label_paths
            or row["content_hash"] in hashes
        ):
            raise ValueError(f"Duplicate path/hash in {expected_split} manifest")
        image_path = _safe_repository_path(root, row["image_path"], "image")
        label_path = _safe_repository_path(root, row["label_path"], "label")
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing {expected_split} image/label pair: {image_path}")
        if image_path.name in basenames:
            raise ValueError("Image basenames must be unique for deterministic prediction mapping")
        record: dict[str, Any] = {**row, "image_absolute": image_path, "label_absolute": label_path}
        if verify_pixels_and_labels:
            actual_hash = sha256_file(image_path)
            if actual_hash != row["content_hash"]:
                raise ValueError(f"Image hash mismatch: {row['image_path']}")
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
            if width != int(row["image_width"]) or height != int(row["image_height"]):
                raise ValueError(f"Image dimension mismatch: {row['image_path']}")
            labels = _parse_labels(label_path, width, height)
            if len(labels) != int(row["object_count"]):
                raise ValueError(f"Object-count mismatch: {row['label_path']}")
            record["ground_truth"] = labels
        image_paths.add(row["image_path"])
        label_paths.add(row["label_path"])
        hashes.add(row["content_hash"])
        basenames.add(image_path.name)
        validated.append(record)
    return validated


def validate_no_split_leakage(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    fields = ("image_path", "label_path", "content_hash", "source_group_id")
    names = tuple(splits)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            for field in fields:
                overlap = {row[field] for row in splits[first]} & {
                    row[field] for row in splits[second]
                }
                if overlap:
                    raise ValueError(f"Leakage between {first}/{second} by {field}")
    return {name: len(rows) for name, rows in splits.items()}


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def validate_git_gate(root: Path, contract_path: Path) -> dict[str, str]:
    if _git(root, "status", "--porcelain"):
        raise ValueError("Tracked/untracked worktree must be clean before the campaign")
    branch = _git(root, "branch", "--show-current")
    if branch != "main":
        raise ValueError("Sprint 5 campaign must run from main")
    head = _git(root, "rev-parse", "HEAD")
    upstream = _git(root, "rev-parse", "origin/main")
    if head != upstream:
        raise ValueError("Local HEAD must equal the fetched origin/main revision")
    contract_commit = _git(root, "log", "-1", "--format=%H", "--", str(contract_path))
    committed_hash = hashlib.sha256(
        subprocess.run(
            ["git", "show", f"{contract_commit}:{contract_path.relative_to(root).as_posix()}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    ).hexdigest()
    if committed_hash != sha256_file(contract_path):
        raise ValueError("Working evaluation contract differs from its committed version")
    return {
        "branch": branch,
        "execution_revision": head,
        "origin_main_revision": upstream,
        "contract_commit": contract_commit,
    }


def _write_dataset_descriptor(output_root: Path, splits: dict[str, list[dict[str, Any]]]) -> Path:
    input_dir = output_root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    lists: dict[str, str] = {}
    for split, rows in splits.items():
        list_path = input_dir / f"{split}_images.txt"
        list_path.write_text(
            "".join(f"{row['image_absolute'].as_posix()}\n" for row in rows), encoding="utf-8"
        )
        lists[split] = list_path.resolve().as_posix()
    descriptor = input_dir / "protected_evaluation_dataset.yaml"
    rootless = output_root.resolve().as_posix()
    descriptor.write_text(
        yaml.safe_dump(
            {
                "path": rootless,
                "train": lists["train"],
                "val": lists["val"],
                "test": lists["test"],
                "names": {index: name for index, name in enumerate(CLASS_NAMES)},
                "nc": len(CLASS_NAMES),
                "synthdet_note": "test list is exclusive to the locked Sprint 5 campaign",
                "synthdet_output_root": rootless,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return descriptor


def _iou_xywh(first: list[float], second: list[float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _size_name(area: float) -> str:
    if area < 1024:
        return "small"
    if area < 9216:
        return "medium"
    return "large"


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def calculate_object_size_metrics(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compute size-stratified AP with deterministic class-aware greedy matching."""

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_image[prediction["file_name"]].append(prediction)
    output: list[dict[str, Any]] = []
    thresholds = np.arange(0.5, 0.96, 0.05)
    for size in SIZE_NAMES:
        ground_truth: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            for truth in record["ground_truth"]:
                if _size_name(truth["area_pixels"]) == size:
                    ground_truth[(record["image_absolute"].name, truth["class_id"])].append(truth)
        class_aps: dict[int, list[float]] = defaultdict(list)
        for class_id in range(len(CLASS_NAMES)):
            total_truth = sum(
                len(items) for (filename, cls), items in ground_truth.items() if cls == class_id
            )
            if total_truth == 0:
                continue
            candidates = sorted(
                (
                    prediction
                    for filename, items in by_image.items()
                    for prediction in items
                    if int(prediction["category_id"]) == class_id
                    and _size_name(float(prediction["bbox"][2]) * float(prediction["bbox"][3]))
                    == size
                ),
                key=lambda item: (-float(item["score"]), item["file_name"], item["bbox"]),
            )
            for threshold in thresholds:
                matched: set[tuple[str, int, int]] = set()
                true_positives: list[int] = []
                false_positives: list[int] = []
                for prediction in candidates:
                    key = (prediction["file_name"], class_id)
                    truths = ground_truth.get(key, [])
                    best_index = -1
                    best_iou = -1.0
                    for truth_index, truth in enumerate(truths):
                        match_key = (prediction["file_name"], class_id, truth_index)
                        if match_key in matched:
                            continue
                        iou = _iou_xywh(prediction["bbox"], truth["bbox_xywh_pixels"])
                        if iou > best_iou:
                            best_iou, best_index = iou, truth_index
                    is_match = best_index >= 0 and best_iou >= threshold
                    true_positives.append(int(is_match))
                    false_positives.append(int(not is_match))
                    if is_match:
                        matched.add((prediction["file_name"], class_id, best_index))
                if candidates:
                    tp = np.cumsum(true_positives)
                    fp = np.cumsum(false_positives)
                    recall = tp / total_truth
                    precision = tp / np.maximum(tp + fp, 1)
                    class_aps[class_id].append(_average_precision(recall, precision))
                else:
                    class_aps[class_id].append(0.0)
        ap50 = [values[0] for values in class_aps.values()]
        ap5095 = [float(np.mean(values)) for values in class_aps.values()]
        output.append(
            {
                "size": size,
                "ground_truth_instances": sum(len(items) for items in ground_truth.values()),
                "classes_with_ground_truth": len(class_aps),
                "map50": float(np.mean(ap50)) if ap50 else None,
                "map50_95": float(np.mean(ap5095)) if ap5095 else None,
                "method": (
                    "101-point interpolated AP; class-aware matching within fixed GT-size stratum"
                ),
            }
        )
    return output


def _per_image_records(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        prediction = dict(prediction)
        class_id = int(prediction["category_id"])
        prediction["class_id"] = class_id
        prediction["class_name"] = CLASS_NAMES[class_id]
        by_image[prediction["file_name"]].append(prediction)
    output: list[dict[str, Any]] = []
    for record in records:
        width = int(record["image_width"])
        height = int(record["image_height"])
        image_predictions = sorted(
            by_image[record["image_absolute"].name], key=lambda item: -float(item["score"])
        )
        for prediction in image_predictions:
            x, y, box_width, box_height = prediction["bbox"]
            prediction["bbox_xywh_normalized"] = [
                x / width,
                y / height,
                box_width / width,
                box_height / height,
            ]
        output.append(
            {
                "image_reference": record["image_path"],
                "image_content_sha256": record["content_hash"],
                "width": width,
                "height": height,
                "ground_truth": record["ground_truth"],
                "predictions": image_predictions,
            }
        )
    return output


def _extract_metrics(metrics: Any) -> tuple[dict[str, float], list[dict[str, Any]]]:
    precision, recall, map50, map50_95 = (float(value) for value in metrics.box.mean_results())
    class_index = {
        int(class_id): index for index, class_id in enumerate(metrics.box.ap_class_index)
    }
    per_class: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        if class_id not in class_index:
            row = {"precision": None, "recall": None, "ap50": None, "ap50_95": None}
        else:
            values = metrics.box.class_result(class_index[class_id])
            row = dict(
                zip(
                    ("precision", "recall", "ap50", "ap50_95"),
                    map(float, values),
                    strict=True,
                )
            )
        per_class.append({"class_id": class_id, "class_name": class_name, **row})
    return (
        {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "macro_per_class_ap50_95": float(
                np.mean([row["ap50_95"] for row in per_class if row["ap50_95"] is not None])
            ),
        },
        per_class,
    )


def _run_model(
    root: Path,
    contract: dict[str, Any],
    regime: str,
    dataset_yaml: Path,
    attempt_root: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluation = contract["evaluation"]
    checkpoint = root / contract["checkpoints"][regime]["path"]
    model = YOLO(str(checkpoint))
    names = [model.names[index] for index in range(len(model.names))]
    if names != CLASS_NAMES:
        raise ValueError(f"{regime} checkpoint class order changed before evaluation")
    metrics = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=evaluation["image_size"],
        batch=evaluation["batch"],
        device=evaluation["device"],
        workers=evaluation["workers"],
        seed=evaluation["seed"],
        deterministic=evaluation["deterministic"],
        conf=evaluation["confidence_threshold"],
        iou=evaluation["iou_threshold"],
        max_det=evaluation["max_detections"],
        augment=evaluation["augment"],
        agnostic_nms=evaluation["agnostic_nms"],
        single_cls=evaluation["single_class"],
        rect=evaluation["rectangular_batches"],
        half=evaluation["half_precision"],
        save_json=evaluation["save_json"],
        save_txt=evaluation["save_txt"],
        save_conf=evaluation["save_confidence"],
        plots=evaluation["plots"],
        project=str(attempt_root),
        name=regime,
        exist_ok=False,
        verbose=True,
    )
    save_dir = Path(metrics.save_dir)
    prediction_path = save_dir / "predictions.json"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"{regime} did not produce predictions.json")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    scalar, per_class = _extract_metrics(metrics)
    speed = {key: float(value) for key, value in metrics.speed.items() if key != "loss"}
    latency = sum(speed.values())
    model_parameters = sum(parameter.numel() for parameter in model.model.parameters())
    try:
        info = model.model.info(verbose=False)
        flops = float(info[3]) if isinstance(info, tuple) and len(info) > 3 else None
    except (AttributeError, IndexError, TypeError, ValueError):
        flops = None
    per_image = _per_image_records(records, predictions)
    per_image_path = save_dir / "per_image_predictions.json"
    _write_json(per_image_path, per_image)
    result = {
        "schema_version": contract["output_schema"]["version"],
        "campaign_id": contract["campaign_id"],
        "regime": regime,
        "real_percentage": REAL_PERCENTAGES[regime],
        "checkpoint": {
            "sha256": contract["checkpoints"][regime]["sha256"],
            "size_bytes": checkpoint.stat().st_size,
            "parameter_count": model_parameters,
            "flops_giga": flops,
        },
        "metrics": scalar,
        "per_class": per_class,
        "object_size_metrics": calculate_object_size_metrics(records, predictions),
        "confusion_matrix": _jsonable(metrics.confusion_matrix.matrix),
        "normalized_confusion_matrix": _jsonable(
            metrics.confusion_matrix.matrix.astype(float)
            / np.maximum(metrics.confusion_matrix.matrix.sum(0, keepdims=True), 1)
        ),
        "curves": {
            name: _jsonable(value)
            for name, value in zip(metrics.curves, metrics.curves_results, strict=True)
        },
        "latency_ms_per_image": {**speed, "total": latency},
        "throughput_images_per_second": 1000.0 / latency if latency > 0 else None,
        "prediction_count": len(predictions),
        "prediction_sha256": sha256_file(per_image_path),
        "ultralytics_prediction_sha256": sha256_file(prediction_path),
        "raw_output_directory": save_dir.relative_to(root).as_posix(),
    }
    result_path = save_dir / "sealed_result.json"
    result["result_sha256"] = _canonical_sha256(result)
    result["result_hash_scope"] = "canonical JSON payload excluding result_sha256/hash_scope"
    _write_json(result_path, result)
    return result


def rank_models(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(result: dict[str, Any]) -> tuple[float, ...]:
        metric = result["metrics"]
        return (
            -metric["map50_95"],
            -metric["map50"],
            -metric["macro_per_class_ap50_95"],
            -metric["recall"],
            result["latency_ms_per_image"]["total"],
            result["checkpoint"]["size_bytes"],
        )

    ranked = sorted(results, key=key)
    return [
        {
            "rank": index,
            "regime": result["regime"],
            "recommended": index == 1,
            **result["metrics"],
            "latency_ms": result["latency_ms_per_image"]["total"],
            "checkpoint_size_bytes": result["checkpoint"]["size_bytes"],
        }
        for index, result in enumerate(ranked, start=1)
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_reports(
    report_dir: Path, results: list[dict[str, Any]], ranking: list[dict[str, Any]]
) -> None:
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    regimes = [result["regime"] for result in results]
    x = np.arange(len(regimes))

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures / name, dpi=180, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(10, 5.5))
    for metric, marker in (("map50_95", "o"), ("map50", "s")):
        plt.plot(
            [REAL_PERCENTAGES[name] for name in regimes],
            [result["metrics"][metric] for result in results],
            marker=marker,
            label=metric,
        )
    plt.xlabel("Real training data (%)")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    save("synthetic_to_real_domain_gap_curve.png")

    for metric, filename, title in (
        ("map50_95", "map50_95_vs_real_percentage.png", "mAP@50-95"),
        ("map50", "map50_vs_real_percentage.png", "mAP@50"),
    ):
        plt.figure(figsize=(9, 5.5))
        plt.plot(
            [REAL_PERCENTAGES[name] for name in regimes],
            [result["metrics"][metric] for result in results],
            marker="o",
        )
        plt.xlabel("Real training data (%)")
        plt.ylabel(title)
        plt.ylim(0, 1)
        plt.grid(alpha=0.25)
        save(filename)

    plt.figure(figsize=(11, 5.5))
    width = 0.2
    for offset, metric in enumerate(("precision", "recall", "map50", "map50_95")):
        plt.bar(
            x + (offset - 1.5) * width, [r["metrics"][metric] for r in results], width, label=metric
        )
    plt.xticks(x, regimes, rotation=20)
    plt.ylim(0, 1)
    plt.legend(ncol=4)
    save("final_metric_comparison.png")

    plt.figure(figsize=(10, 5.5))
    width = 0.35
    plt.bar(
        x - width / 2,
        [result["metrics"]["precision"] for result in results],
        width,
        label="Precision",
    )
    plt.bar(
        x + width / 2,
        [result["metrics"]["recall"] for result in results],
        width,
        label="Recall",
    )
    plt.xticks(x, regimes, rotation=20)
    plt.ylim(0, 1)
    plt.legend()
    save("precision_recall_comparison.png")

    plt.figure(figsize=(10, 5.5))
    plt.bar([row["regime"] for row in ranking], [row["map50_95"] for row in ranking])
    plt.xticks(rotation=20)
    plt.ylim(0, 1)
    plt.ylabel("mAP@50-95")
    save("final_ranking.png")

    plt.figure(figsize=(11, 6))
    width = 0.14
    for index, result in enumerate(results):
        plt.bar(
            np.arange(len(CLASS_NAMES)) + (index - 2) * width,
            [row["ap50_95"] or 0 for row in result["per_class"]],
            width,
            label=result["regime"],
        )
    plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=20)
    plt.ylim(0, 1)
    plt.legend(ncol=3)
    save("per_class_ap_comparison.png")

    plt.figure(figsize=(10, 5.5))
    width = 0.14
    for index, result in enumerate(results):
        plt.bar(
            np.arange(3) + (index - 2) * width,
            [row["map50_95"] or 0 for row in result["object_size_metrics"]],
            width,
            label=result["regime"],
        )
    plt.xticks(np.arange(3), SIZE_NAMES)
    plt.ylim(0, 1)
    plt.legend(ncol=3)
    save("object_size_comparison.png")

    plt.figure(figsize=(10, 5.5))
    plt.bar(regimes, [result["latency_ms_per_image"]["total"] for result in results])
    plt.xticks(rotation=20)
    plt.ylabel("Milliseconds per image (CPU)")
    save("latency_comparison.png")

    radar_metrics = ("precision", "recall", "map50", "map50_95")
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    figure, axis = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for result in results:
        values = [result["metrics"][metric] for metric in radar_metrics]
        values += values[:1]
        axis.plot(angles, values, label=result["regime"])
    axis.set_xticks(angles[:-1], radar_metrics)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    save("radar_comparison.png")

    validation_path = report_dir.parent / "training/sprint4b_v2_validation_summary.csv"
    if validation_path.is_file():
        with validation_path.open(encoding="utf-8", newline="") as handle:
            validation_rows = {row["regime"]: row for row in csv.DictReader(handle)}
        validation_values = []
        for regime in regimes:
            row = validation_rows[regime]
            key = next(
                key
                for key in row
                if key.lower().replace("-", "_") in {"map50_95", "map_50_95", "validation_map50_95"}
            )
            validation_values.append(float(row[key]))
        plt.figure(figsize=(10, 5.5))
        plt.bar(x - width / 2, validation_values, width, label="Validation")
        plt.bar(
            x + width / 2,
            [result["metrics"]["map50_95"] for result in results],
            width,
            label="Protected test",
        )
        plt.xticks(x, regimes, rotation=20)
        plt.ylim(0, 1)
        plt.legend()
        save("validation_vs_test_comparison.png")

    model_figures = figures / "models"
    for result in results:
        source = report_dir.parents[1] / result["raw_output_directory"]
        destination = model_figures / result["regime"]
        destination.mkdir(parents=True, exist_ok=True)
        for filename in (
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxPR_curve.png",
            "BoxF1_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
        ):
            plot = source / filename
            if plot.is_file():
                shutil.copy2(plot, destination / filename)


def _write_reports(
    root: Path, contract: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    report_dir = root / "reports/evaluation"
    ranking = rank_models(results)
    metric_rows = [
        {
            "regime": result["regime"],
            "real_percentage": result["real_percentage"],
            **result["metrics"],
            "prediction_count": result["prediction_count"],
        }
        for result in results
    ]
    _write_csv(report_dir / "final_test_metrics.csv", metric_rows)
    _write_json(
        report_dir / "final_test_metrics.json",
        {"campaign_id": contract["campaign_id"], "models": results},
    )
    _write_csv(report_dir / "final_model_ranking.csv", ranking)
    _write_csv(
        report_dir / "per_class_metrics.csv",
        [{"regime": result["regime"], **row} for result in results for row in result["per_class"]],
    )
    _write_csv(
        report_dir / "object_size_metrics.csv",
        [
            {"regime": result["regime"], **row}
            for result in results
            for row in result["object_size_metrics"]
        ],
    )
    _write_csv(
        report_dir / "latency_metrics.csv",
        [
            {
                "regime": result["regime"],
                **result["latency_ms_per_image"],
                "throughput_images_per_second": result["throughput_images_per_second"],
            }
            for result in results
        ],
    )
    domain_rows = [
        {
            **row,
            "absolute_map50_95_change_from_synthetic_only": row["map50_95"]
            - metric_rows[0]["map50_95"],
            "relative_map50_95_change_from_synthetic_only_percent": (
                (row["map50_95"] / metric_rows[0]["map50_95"] - 1) * 100
                if metric_rows[0]["map50_95"]
                else None
            ),
        }
        for row in metric_rows
    ]
    _write_csv(report_dir / "domain_gap_analysis.csv", domain_rows)
    _plot_reports(report_dir, results, ranking)
    winner = ranking[0]
    ranking_md = [
        "# Final Sprint 5 Model Ranking",
        "",
        "Ranking follows the preregistered rule: test mAP@50-95, then the declared tie-breakers.",
        "",
        "| Rank | Regime | mAP@50-95 | mAP@50 | Precision | Recall | Recommended |",
        "|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for row in ranking:
        recommendation = "yes" if row["recommended"] else "no"
        ranking_md.append(
            f"| {row['rank']} | `{row['regime']}` | {row['map50_95']:.6f} | "
            f"{row['map50']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{recommendation} |"
        )
    (report_dir / "final_model_ranking.md").write_text(
        "\n".join(ranking_md) + "\n", encoding="utf-8"
    )
    report = (
        "# Sprint 5 Final Protected-Test Evaluation\n\n"
        f"Campaign: `{contract['campaign_id']}`. The fixed real Split V2 test set contains "
        "68 images. "
        "All five checkpoints were evaluated once with the shared frozen configuration.\n\n"
        f"The preregistered ranking recommends `{winner['regime']}` with test mAP@50-95 "
        f"{winner['map50_95']:.6f}. This ranking is not based on validation results.\n\n"
        "The test set is small, and Penguin appears in only four test images; class-level and "
        "aggregate conclusions must therefore be interpreted cautiously.\n"
    )
    (report_dir / "sprint5_final_report.md").write_text(report, encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "ultralytics": ultralytics_version,
        "device": contract["evaluation"]["device"],
        "cpu": platform.processor(),
    }
    _write_json(report_dir / "sprint5_environment.json", environment)
    return {"ranking": ranking, "recommended_model": winner["regime"], "environment": environment}


def run_campaign(root: Path, contract_path: Path) -> dict[str, Any]:
    root = root.resolve()
    contract_path = contract_path.resolve()
    validation = validate_contract(root, contract_path)
    git = validate_git_gate(root, contract_path)
    contract = load_contract(contract_path)
    output_root = root / contract["campaign"]["output_root"]
    lock_path = root / "reports/evaluation/sprint5_campaign_lock.json"
    final_metrics_path = root / "reports/evaluation/final_test_metrics.json"
    if output_root.exists() or lock_path.exists() or final_metrics_path.exists():
        raise FileExistsError("Campaign/output already exists; silent overwrite is forbidden")

    attempt_id = "attempt-001"
    lock = {
        "schema_version": 1,
        "status": "locked_preflight",
        "campaign_id": contract["campaign_id"],
        "attempt_id": attempt_id,
        "start_timestamp_utc": _utc_now(),
        "contract_path": contract_path.relative_to(root).as_posix(),
        "contract_sha256": validation["contract_yaml_sha256"],
        **git,
        "exact_command": contract["campaign"]["exact_command"],
        "test_identity": contract["protected_test"],
        "checkpoints": contract["checkpoints"],
        "environment": contract["runtime"],
        "split_preflight": {"status": "pending"},
        "authorized_test_access": {
            "campaign_count": 1,
            "preflight_images_read": "pending_up_to_68",
            "model_evaluation_passes_expected": 5,
        },
        "expected_models": list(REGIMES),
        "expected_output_root": output_root.relative_to(root).as_posix(),
        "expected_outputs": [
            "reports/evaluation/final_test_metrics.csv",
            "reports/evaluation/final_test_metrics.json",
            "reports/evaluation/final_model_ranking.csv",
            "reports/evaluation/final_model_ranking.md",
            "reports/evaluation/per_class_metrics.csv",
            "reports/evaluation/object_size_metrics.csv",
            "reports/evaluation/latency_metrics.csv",
            "reports/evaluation/domain_gap_analysis.csv",
            "reports/evaluation/sprint5_final_report.md",
            "reports/evaluation/sprint5_hash_report.json",
            "reports/evaluation/sprint5_environment.json",
        ],
    }
    _write_json(lock_path, lock)
    manifests = root / "manifests/aquarium/v2"
    try:
        train = validate_manifest_files(
            root,
            manifests / "real_train.csv",
            expected_split="train",
            expected_count=427,
            verify_pixels_and_labels=False,
        )
        val = validate_manifest_files(
            root,
            manifests / "real_val.csv",
            expected_split="val",
            expected_count=140,
            verify_pixels_and_labels=False,
        )
        # This is the first authorized protected pixel/label access in the project.
        test = validate_manifest_files(
            root,
            manifests / "real_test.csv",
            expected_split="test",
            expected_count=68,
            verify_pixels_and_labels=True,
        )
        split_counts = validate_no_split_leakage({"train": train, "val": val, "test": test})
        output_root.mkdir(parents=True, exist_ok=False)
        dataset_yaml = _write_dataset_descriptor(
            output_root, {"train": train, "val": val, "test": test}
        )
    except Exception as error:
        lock.update(
            {
                "status": "failed_preflight",
                "end_timestamp_utc": _utc_now(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "protected_files_read_before_failure": "unknown_up_to_68",
            }
        )
        _write_json(lock_path, lock)
        raise
    lock.update(
        {
            "status": "locked_running",
            "split_preflight": {
                "status": "passed",
                "counts": split_counts,
                "image_hashes_verified": 68,
                "image_decode_verified": 68,
                "label_files_verified": 68,
                "path_hash_source_group_leakage": 0,
            },
            "authorized_test_access": {
                "campaign_count": 1,
                "preflight_images_read": 68,
                "model_evaluation_passes_expected": 5,
            },
        }
    )
    _write_json(lock_path, lock)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ["PYTHONHASHSEED"] = "42"
    results: list[dict[str, Any]] = []
    attempt_root = output_root / attempt_id
    try:
        for regime in REGIMES:
            results.append(_run_model(root, contract, regime, dataset_yaml, attempt_root, test))
        report_summary = _write_reports(root, contract, results)
        hash_report = {
            "campaign_id": contract["campaign_id"],
            "contract_sha256": validation["contract_yaml_sha256"],
            "model_results": {
                result["regime"]: {
                    "checkpoint_sha256": result["checkpoint"]["sha256"],
                    "prediction_sha256": result["prediction_sha256"],
                    "ultralytics_prediction_sha256": result["ultralytics_prediction_sha256"],
                    "result_sha256": result["result_sha256"],
                }
                for result in results
            },
        }
        _write_json(root / "reports/evaluation/sprint5_hash_report.json", hash_report)
    except Exception as error:
        lock.update(
            {
                "status": "failed_technical_attempt",
                "end_timestamp_utc": _utc_now(),
                "completed_models": [result["regime"] for result in results],
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "partial_metrics_for_tuning_forbidden": True,
            }
        )
        _write_json(lock_path, lock)
        raise
    lock.update(
        {
            "status": "completed_sealed",
            "end_timestamp_utc": _utc_now(),
            "completed_models": list(REGIMES),
            "successful_campaign_count": 1,
            "model_evaluation_passes": 5,
            "recommended_model": report_summary["recommended_model"],
            "results_sealed": True,
        }
    )
    _write_json(lock_path, lock)
    return {"lock": lock, **report_summary, "results": results}
