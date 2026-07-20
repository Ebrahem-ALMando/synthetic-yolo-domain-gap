from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from synthdet.evaluation.campaign import (
    _normalize_ultralytics_predictions,
    calculate_object_size_metrics,
    rank_models,
    validate_manifest_files,
    validate_no_split_leakage,
)


def _manifest_row(root: Path, split: str, name: str, group: str) -> dict[str, str]:
    image = root / f"datasets/{split}/images/{name}.jpg"
    label = root / f"datasets/{split}/labels/{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    colors = {"train": "blue", "val": "green", "test": "red"}
    Image.new("RGB", (100, 80), colors[split]).save(image)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return {
        "image_path": image.relative_to(root).as_posix(),
        "label_path": label.relative_to(root).as_posix(),
        "content_hash": hashlib.sha256(image.read_bytes()).hexdigest(),
        "source_group_id": group,
        "image_width": "100",
        "image_height": "80",
        "classes_present": "fish",
        "object_count": "1",
        "split": split,
        "inclusion_status": "included",
    }


def _write_manifest(path: Path, row: dict[str, str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def test_manifest_pixel_label_integrity_and_leakage(tmp_path: Path) -> None:
    splits = {}
    for split in ("train", "val", "test"):
        row = _manifest_row(tmp_path, split, split, f"group-{split}")
        path = tmp_path / f"{split}.csv"
        _write_manifest(path, row)
        splits[split] = validate_manifest_files(
            tmp_path,
            path,
            expected_split=split,
            expected_count=1,
            verify_pixels_and_labels=True,
        )
    assert validate_no_split_leakage(splits) == {"train": 1, "val": 1, "test": 1}
    assert splits["test"][0]["ground_truth"][0]["class_name"] == "fish"


def test_manifest_rejects_image_hash_change(tmp_path: Path) -> None:
    row = _manifest_row(tmp_path, "test", "sample", "group-test")
    row["content_hash"] = "0" * 64
    path = tmp_path / "test.csv"
    _write_manifest(path, row)
    with pytest.raises(ValueError, match="Image hash mismatch"):
        validate_manifest_files(
            tmp_path,
            path,
            expected_split="test",
            expected_count=1,
            verify_pixels_and_labels=True,
        )


def test_object_size_metrics_and_predeclared_ranking() -> None:
    records = [
        {
            "image_absolute": Path("sample.jpg"),
            "ground_truth": [
                {
                    "class_id": 0,
                    "bbox_xywh_pixels": [10.0, 10.0, 20.0, 20.0],
                    "area_pixels": 400.0,
                }
            ],
        }
    ]
    predictions = [
        {
            "file_name": "sample.jpg",
            "category_id": 0,
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "score": 0.9,
        }
    ]
    sizes = calculate_object_size_metrics(records, predictions)
    assert sizes[0]["map50_95"] == pytest.approx(0.995)
    assert sizes[1]["map50_95"] is None

    def result(regime: str, score: float) -> dict:
        return {
            "regime": regime,
            "metrics": {
                "map50_95": score,
                "map50": score,
                "macro_per_class_ap50_95": score,
                "recall": score,
            },
            "latency_ms_per_image": {"total": 10.0},
            "checkpoint": {"size_bytes": 100},
        }

    ranking = rank_models([result("real_only", 0.7), result("synthetic_only", 0.8)])
    assert ranking[0]["regime"] == "synthetic_only"
    assert ranking[0]["recommended"] is True


def test_ultralytics_one_based_json_categories_are_normalized() -> None:
    predictions = [{"category_id": 1}, {"category_id": 7}]
    normalized = _normalize_ultralytics_predictions(predictions, 7)
    assert [row["category_id"] for row in normalized] == [0, 6]
    assert [row["ultralytics_category_id"] for row in normalized] == [1, 7]
