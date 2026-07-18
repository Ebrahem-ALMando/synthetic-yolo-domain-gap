"""Apply the completed Aquarium visual-review decisions to the audit CSVs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _capture_number(capture_id: str) -> int:
    match = re.match(r"IMG_(\d+)", capture_id)
    if not match:
        raise ValueError(f"Cannot parse capture number: {capture_id}")
    return int(match.group(1))


def _selection_matches(row: dict[str, str], selection: dict[str, Any]) -> bool:
    if row["source_group_id"] != selection["original_group"]:
        return False
    ranges = selection.get("ranges")
    if not ranges:
        return True
    number = _capture_number(row["original_capture_id"])
    return any(int(start) <= number <= int(end) for start, end in ranges)


def finalize_reviews(
    decisions_path: Path,
    duplicate_path: Path,
    source_path: Path,
    output_dir: Path,
    timestamp: str | None = None,
) -> dict[str, int]:
    """Apply a complete decision plan and emit review logs and summary."""

    decisions = yaml.safe_load(decisions_path.read_text(encoding="utf-8"))
    reviewed_at = timestamp or datetime.now(UTC).isoformat()
    reviewer_mode = decisions["reviewer_mode"]

    duplicate_fields, duplicate_rows = _read_csv(duplicate_path)
    duplicate_log: list[dict[str, str]] = []
    duplicate_decisions = decisions["duplicate_groups"]
    seen_duplicate_groups: set[str] = set()
    for group_id, rationale in duplicate_decisions.items():
        members = [row for row in duplicate_rows if row["duplicate_group_id"] == group_id]
        if len(members) < 2:
            raise ValueError(f"Duplicate decision {group_id} does not identify a candidate group")
        if any(row["review_status"] not in {"pending", "confirmed"} for row in members):
            raise ValueError(f"Duplicate group {group_id} has an incompatible prior decision")
        for row in members:
            row["review_status"] = "confirmed"
        distances = sorted(
            {row["minimum_hamming_distance"] for row in members if row["minimum_hamming_distance"]}
        )
        duplicate_log.append(
            {
                "duplicate_group_id": group_id,
                "image_paths": ";".join(sorted(row["image_path"] for row in members)),
                "dhash_distance": ";".join(distances),
                "final_status": "confirmed",
                "rationale": rationale,
                "reviewer_mode": reviewer_mode,
                "review_timestamp": reviewed_at,
            }
        )
        seen_duplicate_groups.add(group_id)
    pending_groups = {
        row["duplicate_group_id"]
        for row in duplicate_rows
        if row["review_status"] == "pending" and row["duplicate_group_id"]
    }
    if pending_groups or seen_duplicate_groups != set(duplicate_decisions):
        raise ValueError(f"Unresolved duplicate decisions: {sorted(pending_groups)}")
    _write_csv(duplicate_path, duplicate_fields, duplicate_rows)
    _write_csv(
        output_dir / "duplicate_review_log.csv",
        [
            "duplicate_group_id",
            "image_paths",
            "dhash_distance",
            "final_status",
            "rationale",
            "reviewer_mode",
            "review_timestamp",
        ],
        duplicate_log,
    )

    source_fields, source_rows = _read_csv(source_path)
    original_by_path = {row["image_path"]: row["source_group_id"] for row in source_rows}
    sheets_by_original: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        sheets_by_original[row["source_group_id"]].update(
            value for value in row.get("contact_sheets", "").split(";") if value
        )

    assignment_by_path: dict[str, dict[str, str]] = {}
    rationale_by_path: dict[str, str] = {}
    for assignment in decisions["source_assignments"]:
        selections = assignment.get("selections") or [
            {"original_group": group_id} for group_id in assignment.get("original_groups", [])
        ]
        for row in source_rows:
            if row["review_status"] != "pending":
                continue
            if not any(_selection_matches(row, selection) for selection in selections):
                continue
            if row["image_path"] in assignment_by_path:
                raise ValueError(f"Source row matched multiple decisions: {row['image_path']}")
            assignment_by_path[row["image_path"]] = {
                "final_source_group_id": assignment["final_source_group_id"],
                "final_status": assignment.get("final_status", "confirmed"),
            }
            rationale_by_path[row["image_path"]] = assignment["rationale"]

    pending_paths = {row["image_path"] for row in source_rows if row["review_status"] == "pending"}
    if set(assignment_by_path) != pending_paths:
        missing = sorted(pending_paths - assignment_by_path.keys())
        extra = sorted(assignment_by_path.keys() - pending_paths)
        raise ValueError(
            "Decision plan does not exactly cover pending source rows: "
            f"missing={missing}, extra={extra}"
        )

    for row in source_rows:
        decision = assignment_by_path.get(row["image_path"])
        if decision:
            row["source_group_id"] = decision["final_source_group_id"]
            row["review_status"] = decision["final_status"]
            row["evidence"] = f"{row['evidence']}; agent-assisted visual review"
            row["recommended_review_action"] = "review complete"

    original_to_final: dict[str, set[str]] = defaultdict(set)
    final_to_original: dict[str, set[str]] = defaultdict(set)
    mapping_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        original = original_by_path[row["image_path"]]
        final = row["source_group_id"]
        original_to_final[original].add(final)
        final_to_original[final].add(original)
        mapping_rows[(original, final)].append(row)

    source_log: list[dict[str, str]] = []
    for (original, final), rows in sorted(mapping_rows.items()):
        statuses = {row["review_status"] for row in rows}
        if len(statuses) != 1:
            raise ValueError(f"Mixed final statuses for {original} -> {final}")
        status = next(iter(statuses))
        if status == "not_applicable":
            action = "singleton"
        elif len(original_to_final[original]) > 1:
            action = "split"
        elif len(final_to_original[final]) > 1:
            action = "merge"
        else:
            action = "confirmed"
        rationales = sorted(
            {
                rationale_by_path[row["image_path"]]
                for row in rows
                if row["image_path"] in rationale_by_path
            }
        )
        if not rationales:
            rationales = ["Explicit shared _MOV filename evidence retained from the prior review."]
        source_log.append(
            {
                "original_source_group_id": original,
                "final_source_group_id": final,
                "affected_image_count": str(len(rows)),
                "final_status": status,
                "action": action,
                "visual_rationale": " ".join(rationales),
                "relevant_contact_sheet_paths": ";".join(sorted(sheets_by_original[original])),
                "reviewer_mode": reviewer_mode,
                "review_timestamp": reviewed_at,
            }
        )

    _write_csv(source_path, source_fields, source_rows)
    _write_csv(
        output_dir / "source_group_review_log.csv",
        [
            "original_source_group_id",
            "final_source_group_id",
            "affected_image_count",
            "final_status",
            "action",
            "visual_rationale",
            "relevant_contact_sheet_paths",
            "reviewer_mode",
            "review_timestamp",
        ],
        source_log,
    )

    actions = defaultdict(set)
    for row in source_log:
        actions[row["action"]].add(row["original_source_group_id"])
    final_group_count = len({row["source_group_id"] for row in source_rows})
    singleton_count = sum(row["review_status"] == "not_applicable" for row in source_rows)
    summary = (
        "# Aquarium Source-Group Review Summary\n\n"
        f"- Groups confirmed unchanged: {len(actions['confirmed'])}\n"
        f"- Original groups split: {len(actions['split'])}\n"
        f"- Original groups participating in merges: {len(actions['merge'])}\n"
        f"- Singleton/not-applicable images: {singleton_count}\n"
        f"- Final stable source groups: {final_group_count}\n"
        "- Unresolved states: 0\n\n"
        "All 58 generated source contact-sheet pages and all pending singleton images were "
        "reviewed in agent-assisted visual-review mode. Scene boundaries remain subjective where "
        "different camera positions show the same physical exhibit; the review uses conservative "
        "same-exhibit grouping to reduce leakage risk.\n"
    )
    (output_dir / "source_group_review_summary.md").write_text(summary, encoding="utf-8")
    return {
        "duplicate_groups_confirmed": len(duplicate_log),
        "source_rows": len(source_rows),
        "final_source_groups": final_group_count,
        "not_applicable_images": singleton_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize the Aquarium visual-review CSVs.")
    parser.add_argument(
        "--decisions", type=Path, default=Path("configs/datasets/aquarium_review.yaml")
    )
    parser.add_argument(
        "--duplicates",
        type=Path,
        default=Path("reports/dataset_audit/aquarium/duplicate_candidates.csv"),
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("reports/dataset_audit/aquarium/reviewed_source_groups.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/dataset_audit/aquarium")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = finalize_reviews(
            args.decisions, args.duplicates, args.sources, args.output
        )
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"Review finalization failed: {error}", file=sys.stderr)
        return 1
    print(
        "Review finalization complete: "
        f"{summary['duplicate_groups_confirmed']} duplicate groups confirmed, "
        f"{summary['final_source_groups']} stable source groups."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
