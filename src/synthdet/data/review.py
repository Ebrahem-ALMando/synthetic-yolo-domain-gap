"""Hard-fail integrity checks for finalized duplicate and source reviews."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required review input not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_review_integrity(
    records_path: Path,
    duplicates_path: Path,
    source_groups_path: Path,
    dataset_root: Path,
) -> list[str]:
    """Return every violation that would make reviewed splitting unsafe."""

    records = _read_csv(records_path)
    duplicate_rows = _read_csv(duplicates_path)
    source_rows = _read_csv(source_groups_path)
    included = {row["image_path"] for row in records if row["inclusion_status"] == "included"}
    excluded = {row["image_path"] for row in records if row["inclusion_status"] != "included"}
    errors: list[str] = []

    allowed_duplicate_statuses = {"confirmed", "rejected", "not_applicable"}
    duplicate_statuses = {row.get("review_status", "") for row in duplicate_rows}
    invalid_duplicate_statuses = sorted(duplicate_statuses - allowed_duplicate_statuses)
    if "pending" in duplicate_statuses:
        errors.append("Pending duplicate review remains")
    if invalid_duplicate_statuses:
        errors.append(
            "Invalid duplicate review status: " + ", ".join(invalid_duplicate_statuses)
        )

    duplicate_counts = Counter(row.get("image_path", "") for row in duplicate_rows)
    duplicated_duplicate_paths = sorted(
        path for path, count in duplicate_counts.items() if count > 1
    )
    if duplicated_duplicate_paths:
        errors.append(
            "Duplicate analysis assigns an image more than once: "
            + ", ".join(duplicated_duplicate_paths)
        )
    duplicate_paths = set(duplicate_counts)
    if duplicate_paths != included:
        missing = sorted(included - duplicate_paths)
        extra = sorted(duplicate_paths - included)
        if missing:
            errors.append("Duplicate review is missing accepted images: " + ", ".join(missing))
        if extra:
            errors.append("Duplicate review includes non-accepted images: " + ", ".join(extra))

    confirmed_duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for row in duplicate_rows:
        status = row.get("review_status", "")
        group_id = row.get("duplicate_group_id", "").strip()
        if status == "confirmed":
            if not group_id:
                errors.append(
                    "Confirmed duplicate has no group identity: "
                    f"{row.get('image_path', '')}"
                )
            else:
                confirmed_duplicate_groups[group_id].append(row.get("image_path", ""))
        elif status in {"rejected", "not_applicable"} and group_id:
            errors.append(
                f"{status} duplicate row retains a group identity: {row.get('image_path', '')}"
            )
    for group_id, members in sorted(confirmed_duplicate_groups.items()):
        if len(members) < 2:
            errors.append(f"Confirmed duplicate group has fewer than two images: {group_id}")

    allowed_source_statuses = {"confirmed", "not_applicable"}
    source_statuses = {row.get("review_status", "") for row in source_rows}
    if "pending" in source_statuses:
        errors.append("Pending source-group review remains")
    if "split_required" in source_statuses:
        errors.append("Unresolved split_required source-group review remains")
    if "merge_required" in source_statuses:
        errors.append("Unresolved merge_required source-group review remains")
    invalid_source_statuses = sorted(source_statuses - allowed_source_statuses)
    if invalid_source_statuses:
        errors.append("Invalid source review status: " + ", ".join(invalid_source_statuses))

    source_counts = Counter(row.get("image_path", "") for row in source_rows)
    duplicate_source_paths = sorted(path for path, count in source_counts.items() if count > 1)
    if duplicate_source_paths:
        errors.append(
            "Source review assigns an image more than once: " + ", ".join(duplicate_source_paths)
        )
    source_paths = set(source_counts)
    missing_sources = sorted(included - source_paths)
    extra_sources = sorted(source_paths - included)
    if missing_sources:
        errors.append("Source review is missing accepted images: " + ", ".join(missing_sources))
    if extra_sources:
        errors.append("Source review includes non-accepted images: " + ", ".join(extra_sources))
    excluded_sources = sorted(excluded & source_paths)
    if excluded_sources:
        errors.append("Excluded images appear in source review: " + ", ".join(excluded_sources))

    source_by_image: dict[str, str] = {}
    for row in source_rows:
        image_path = row.get("image_path", "")
        source_group_id = row.get("source_group_id", "").strip()
        if not source_group_id:
            errors.append(f"Source review has an empty stable identity: {image_path}")
        source_by_image[image_path] = source_group_id
        if image_path in included and not (dataset_root / image_path).is_file():
            errors.append(f"Reviewed accepted image does not exist: {image_path}")

    for group_id, members in sorted(confirmed_duplicate_groups.items()):
        source_ids = {source_by_image.get(member, "") for member in members}
        if len(source_ids) != 1 or "" in source_ids:
            errors.append(
                f"Confirmed duplicate group is incompatible with source assignments: {group_id}"
            )
    return errors


def require_valid_review(
    records_path: Path,
    duplicates_path: Path,
    source_groups_path: Path,
    dataset_root: Path,
) -> None:
    """Raise one actionable error when the completed review is not safe."""

    errors = validate_review_integrity(
        records_path, duplicates_path, source_groups_path, dataset_root
    )
    if errors:
        raise ValueError("Review integrity validation failed:\n- " + "\n- ".join(errors))
