from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from synthdet.data.acquisition import export_api_url
from synthdet.data.audit import generate_dataset_audit
from synthdet.data.duplicates import analyze_duplicates
from synthdet.data.leakage import validate_leakage
from synthdet.data.review import validate_review_integrity
from synthdet.data.source_groups import propose_source_groups
from synthdet.data.splitting import (
    _manifest_path,
    create_real_splits,
    verify_real_split_reproduction,
)
from synthdet.data.validation import validate_yolo_dataset, write_validation_outputs


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _create_dataset(root: Path) -> None:
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("names: [fish, shark]\n", encoding="utf-8")
    Image.new("RGB", (100, 80), "navy").save(root / "train" / "images" / "valid.jpg")
    (root / "train" / "labels" / "valid.txt").write_text(
        "0 0.5 0.5 0.2 0.25\n", encoding="utf-8"
    )


def test_validation_and_audit_use_inspected_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw"
    _create_dataset(dataset_root)

    classes, records, issues = validate_yolo_dataset(dataset_root)

    assert classes == ["fish", "shark"]
    assert len(records) == 1
    assert records[0].inclusion_status == "included"
    assert records[0].object_count == 1
    assert records[0].aspect_ratio == 1.25
    assert issues == []

    validation_dir = tmp_path / "validation"
    write_validation_outputs(validation_dir, classes, records, issues)
    statistics = generate_dataset_audit(validation_dir, tmp_path / "audit")
    assert statistics["inspected_images"] == 1
    assert statistics["per_class_object_counts"] == {"fish": 1, "shark": 0}


def test_invalid_annotations_are_excluded_not_repaired(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw"
    _create_dataset(dataset_root)
    (dataset_root / "train" / "labels" / "valid.txt").write_text(
        "7 0.5 0.5 0.2 0.2\n0 0.95 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    _, records, issues = validate_yolo_dataset(dataset_root)

    assert records[0].inclusion_status == "excluded"
    assert records[0].object_count == 0
    assert {issue.code for issue in issues} == {
        "unknown_class",
        "box_outside_image",
        "no_valid_objects",
    }


def test_corrupt_images_and_orphan_labels_remain_traceable(tmp_path: Path) -> None:
    dataset_root = tmp_path / "raw"
    _create_dataset(dataset_root)
    (dataset_root / "train" / "images" / "corrupt.jpg").write_bytes(b"not-an-image")
    (dataset_root / "train" / "labels" / "corrupt.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset_root / "train" / "labels" / "orphan.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    _, records, issues = validate_yolo_dataset(dataset_root)

    corrupt = next(record for record in records if record.image_path.endswith("corrupt.jpg"))
    assert corrupt.inclusion_status == "excluded"
    assert corrupt.exclusion_reasons == ["corrupt_image"]
    assert {issue.code for issue in issues} == {"corrupt_image", "orphan_label"}


def test_duplicate_analysis_groups_exact_and_near_candidates(tmp_path: Path) -> None:
    records_path = tmp_path / "records.csv"
    fields = [
        "image_path",
        "content_hash",
        "perceptual_hash",
        "inclusion_status",
    ]
    _write_csv(
        records_path,
        fields,
        [
            {
                "image_path": "a.jpg",
                "content_hash": "same",
                "perceptual_hash": "0000000000000000",
                "inclusion_status": "included",
            },
            {
                "image_path": "b.jpg",
                "content_hash": "same",
                "perceptual_hash": "0000000000000000",
                "inclusion_status": "included",
            },
            {
                "image_path": "c.jpg",
                "content_hash": "different",
                "perceptual_hash": "0000000000000001",
                "inclusion_status": "included",
            },
        ],
    )
    output = tmp_path / "duplicates.csv"

    assert analyze_duplicates(records_path, output, threshold=1) == 1
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len({row["duplicate_group_id"] for row in rows}) == 1
    assert all(row["review_status"] == "pending" for row in rows)


def _split_inputs(root: Path) -> tuple[Path, Path, Path]:
    records_path = root / "records.csv"
    duplicate_path = root / "duplicates.csv"
    source_path = root / "sources.csv"
    record_fields = [
        "image_path",
        "label_path",
        "content_hash",
        "perceptual_hash",
        "width",
        "height",
        "classes_present",
        "object_count",
        "inclusion_status",
        "exclusion_reasons",
    ]
    record_rows = []
    duplicate_rows = []
    source_rows = []
    for index in range(20):
        image_path = f"images/image-{index:02d}.jpg"
        record_rows.append(
            {
                "image_path": image_path,
                "label_path": f"labels/image-{index:02d}.txt",
                "content_hash": f"sha-{index:02d}",
                "perceptual_hash": f"{index:016x}",
                "width": "100",
                "height": "80",
                "classes_present": "fish" if index % 2 == 0 else "shark",
                "object_count": "1",
                "inclusion_status": "included",
                "exclusion_reasons": "",
            }
        )
        duplicate_rows.append(
            {
                "image_path": image_path,
                "content_hash": f"sha-{index:02d}",
                "perceptual_hash": f"{index:016x}",
                "duplicate_group_id": "",
                "match_type": "unique",
                "minimum_hamming_distance": "",
                "review_status": "not_applicable",
            }
        )
        source_rows.append(
            {
                "image_path": image_path,
                "source_group_id": f"source-{index:02d}",
                "review_status": "confirmed",
            }
        )
    _write_csv(records_path, record_fields, record_rows)
    _write_csv(duplicate_path, list(duplicate_rows[0]), duplicate_rows)
    _write_csv(source_path, ["image_path", "source_group_id", "review_status"], source_rows)
    return records_path, duplicate_path, source_path


def test_split_is_deterministic_frozen_and_leakage_free(tmp_path: Path) -> None:
    records, duplicates, sources = _split_inputs(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = create_real_splits(records, duplicates, first_dir, sources)
    second = create_real_splits(records, duplicates, second_dir, sources)

    assert first["combined_split_sha256"] == second["combined_split_sha256"]
    assert all(first["actual_counts"][split] > 0 for split in ("train", "val", "test"))
    assert validate_leakage(first_dir) == []
    reproduced = verify_real_split_reproduction(
        first_dir,
        records_path=records,
        duplicates_path=duplicates,
        source_groups_path=sources,
    )
    assert reproduced["combined_split_sha256"] == first["combined_split_sha256"]


def test_manifest_paths_are_repository_relative() -> None:
    assert _manifest_path(
        Path("datasets/raw/aquarium/export"), "train/images/example.jpg"
    ) == "datasets/raw/aquarium/export/train/images/example.jpg"


def test_split_rejects_pending_source_review(tmp_path: Path) -> None:
    records, duplicates, sources = _split_inputs(tmp_path)
    with sources.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["review_status"] = "pending"
    _write_csv(sources, list(rows[0]), rows)

    with pytest.raises(ValueError, match="Source groups require human review"):
        create_real_splits(records, duplicates, tmp_path / "blocked", sources)


def test_leakage_checker_detects_test_hash_in_future_source(tmp_path: Path) -> None:
    records, duplicates, sources = _split_inputs(tmp_path)
    manifest_dir = tmp_path / "manifests"
    create_real_splits(records, duplicates, manifest_dir, sources)
    with (manifest_dir / "real_test.csv").open(encoding="utf-8", newline="") as handle:
        test_row = next(csv.DictReader(handle))
    future = tmp_path / "synthetic_sources.csv"
    _write_csv(
        future,
        ["image_path", "content_hash"],
        [{"image_path": "different.jpg", "content_hash": test_row["content_hash"]}],
    )

    errors = validate_leakage(manifest_dir, synthetic_source_manifests=[future])

    assert any("Test content hash" in error for error in errors)


def test_acquisition_url_is_official_and_credential_free() -> None:
    url = export_api_url("brad-dwyer", "aquarium-combined", 2, "yolov5pytorch")

    assert url == "https://api.roboflow.com/brad-dwyer/aquarium-combined/2/yolov5pytorch"
    assert "api_key" not in url


def test_source_proposals_confirm_video_and_hold_numbered_run(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image_paths = [
        "train/images/IMG_100_MOV-0_jpg.rf.aaa.jpg",
        "valid/images/IMG_100_MOV-1_jpg.rf.bbb.jpg",
        "train/images/IMG_200_jpg.rf.ccc.jpg",
        "test/images/IMG_201_jpg.rf.ddd.jpg",
    ]
    for index, image_path in enumerate(image_paths):
        target = dataset_root / image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), (index * 40, 10, 20)).save(target)
    records = tmp_path / "records.csv"
    duplicates = tmp_path / "duplicates.csv"
    _write_csv(
        records,
        ["image_path", "inclusion_status"],
        [
            {"image_path": image_path, "inclusion_status": "included"}
            for image_path in image_paths
        ],
    )
    _write_csv(
        duplicates,
        ["image_path", "duplicate_group_id"],
        [{"image_path": image_path, "duplicate_group_id": ""} for image_path in image_paths],
    )
    output = tmp_path / "reviewed_source_groups.csv"

    summary = propose_source_groups(
        records, duplicates, dataset_root, output, tmp_path / "contact_sheets"
    )

    assert summary["proposed_source_groups"] == 2
    assert summary["confirmed_video_groups"] == 1
    assert summary["pending_groups"] == 1
    with output.open(encoding="utf-8", newline="") as handle:
        statuses = {row["review_status"] for row in csv.DictReader(handle)}
    assert statuses == {"confirmed", "pending"}


def _completed_review_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    dataset_root = root / "dataset"
    records = root / "records.csv"
    duplicates = root / "duplicates.csv"
    sources = root / "sources.csv"
    record_rows = []
    duplicate_rows = []
    source_rows = []
    for index in range(3):
        image_path = f"images/image-{index}.jpg"
        target = dataset_root / image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"image")
        record_rows.append(
            {"image_path": image_path, "inclusion_status": "included"}
        )
        duplicate_rows.append(
            {
                "image_path": image_path,
                "duplicate_group_id": "duplicate-1" if index < 2 else "",
                "review_status": "confirmed" if index < 2 else "not_applicable",
            }
        )
        source_rows.append(
            {
                "image_path": image_path,
                "source_group_id": "source-shared" if index < 2 else "source-single",
                "review_status": "confirmed",
            }
        )
    record_rows.append(
        {"image_path": "images/excluded.jpg", "inclusion_status": "excluded"}
    )
    _write_csv(records, ["image_path", "inclusion_status"], record_rows)
    _write_csv(
        duplicates,
        ["image_path", "duplicate_group_id", "review_status"],
        duplicate_rows,
    )
    _write_csv(
        sources,
        ["image_path", "source_group_id", "review_status"],
        source_rows,
    )
    return records, duplicates, sources, dataset_root


def _rewrite_row_status(path: Path, status: str) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["review_status"] = status
    _write_csv(path, list(rows[0]), rows)


def test_review_integrity_accepts_completed_clean_review(tmp_path: Path) -> None:
    records, duplicates, sources, dataset_root = _completed_review_inputs(tmp_path)

    assert validate_review_integrity(records, duplicates, sources, dataset_root) == []


def test_review_integrity_rejects_pending_duplicate(tmp_path: Path) -> None:
    records, duplicates, sources, dataset_root = _completed_review_inputs(tmp_path)
    _rewrite_row_status(duplicates, "pending")

    errors = validate_review_integrity(records, duplicates, sources, dataset_root)

    assert any("Pending duplicate" in error for error in errors)


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("pending", "Pending source-group"),
        ("split_required", "split_required"),
        ("merge_required", "merge_required"),
    ],
)
def test_review_integrity_rejects_unresolved_source_status(
    tmp_path: Path, status: str, message: str
) -> None:
    records, duplicates, sources, dataset_root = _completed_review_inputs(tmp_path)
    _rewrite_row_status(sources, status)

    errors = validate_review_integrity(records, duplicates, sources, dataset_root)

    assert any(message in error for error in errors)


def test_review_integrity_rejects_duplicate_image_assignment(tmp_path: Path) -> None:
    records, duplicates, sources, dataset_root = _completed_review_inputs(tmp_path)
    with sources.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.append(dict(rows[0]))
    _write_csv(sources, list(rows[0]), rows)

    errors = validate_review_integrity(records, duplicates, sources, dataset_root)

    assert any("assigns an image more than once" in error for error in errors)


def test_review_integrity_rejects_incompatible_confirmed_duplicate_group(
    tmp_path: Path,
) -> None:
    records, duplicates, sources, dataset_root = _completed_review_inputs(tmp_path)
    with sources.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["source_group_id"] = "different-source"
    _write_csv(sources, list(rows[0]), rows)

    errors = validate_review_integrity(records, duplicates, sources, dataset_root)

    assert any("incompatible with source assignments" in error for error in errors)


def test_review_integrity_rejects_missing_accepted_image(tmp_path: Path) -> None:
    records, duplicates, sources, dataset_root = _completed_review_inputs(tmp_path)
    with sources.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _write_csv(sources, list(rows[0]), rows[:-1])

    errors = validate_review_integrity(records, duplicates, sources, dataset_root)

    assert any("missing accepted images" in error for error in errors)


def test_review_integrity_rejects_excluded_image(tmp_path: Path) -> None:
    records, duplicates, sources, dataset_root = _completed_review_inputs(tmp_path)
    with sources.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.append(
        {
            "image_path": "images/excluded.jpg",
            "source_group_id": "source-excluded",
            "review_status": "confirmed",
        }
    )
    _write_csv(sources, list(rows[0]), rows)

    errors = validate_review_integrity(records, duplicates, sources, dataset_root)

    assert any("Excluded images" in error for error in errors)
