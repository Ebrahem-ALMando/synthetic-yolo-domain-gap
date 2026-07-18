"""Exact SHA-256 and near-duplicate dHash analysis."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from synthdet.data.hashing import hamming_distance


class _DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def analyze_duplicates(records_path: Path, output_path: Path, threshold: int = 6) -> int:
    """Write duplicate candidates and return the number of multi-image groups."""

    if not 0 <= threshold <= 64:
        raise ValueError("dHash threshold must be between 0 and 64")
    if not records_path.is_file():
        raise FileNotFoundError(f"Image records not found: {records_path}")
    with records_path.open(encoding="utf-8", newline="") as handle:
        records = [
            row for row in csv.DictReader(handle) if row["inclusion_status"] == "included"
        ]
    paths = sorted(row["image_path"] for row in records)
    if not paths:
        raise ValueError("No included image records are available for duplicate analysis")
    by_path = {row["image_path"]: row for row in records}
    groups = _DisjointSet(paths)
    exact_pairs: set[tuple[str, str]] = set()
    near_pairs: set[tuple[str, str]] = set()
    by_sha: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_sha[row["content_hash"]].append(row["image_path"])
    for matches in by_sha.values():
        for index, first in enumerate(sorted(matches)):
            for second in sorted(matches)[index + 1 :]:
                groups.union(first, second)
                exact_pairs.add((first, second))
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            if (first, second) in exact_pairs:
                continue
            distance = hamming_distance(
                by_path[first]["perceptual_hash"], by_path[second]["perceptual_hash"]
            )
            if distance <= threshold:
                groups.union(first, second)
                near_pairs.add((first, second))

    components: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        components[groups.find(path)].append(path)
    duplicate_components = sorted(
        (sorted(members) for members in components.values() if len(members) > 1),
        key=lambda members: members[0],
    )
    group_for_path = {
        path: f"duplicate-{index:04d}"
        for index, members in enumerate(duplicate_components, start=1)
        for path in members
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "content_hash",
        "perceptual_hash",
        "duplicate_group_id",
        "match_type",
        "minimum_hamming_distance",
        "review_status",
        "recommended_review_action",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in paths:
            group_id = group_for_path.get(path, "")
            component = next(
                (members for members in duplicate_components if path in members), [path]
            )
            distances = [
                hamming_distance(
                    by_path[path]["perceptual_hash"], by_path[other]["perceptual_hash"]
                )
                for other in component
                if other != path
            ]
            has_exact = any(
                tuple(sorted((path, other))) in exact_pairs for other in component if other != path
            )
            has_near = any(
                tuple(sorted((path, other))) in near_pairs for other in component if other != path
            )
            if has_exact and has_near:
                match_type = "exact+near"
            else:
                match_type = "exact" if has_exact else "near"
            writer.writerow(
                {
                    "image_path": path,
                    "content_hash": by_path[path]["content_hash"],
                    "perceptual_hash": by_path[path]["perceptual_hash"],
                    "duplicate_group_id": group_id,
                    "match_type": match_type if group_id else "unique",
                    "minimum_hamming_distance": min(distances) if distances else "",
                    "review_status": "pending" if group_id else "not_applicable",
                    "recommended_review_action": (
                        "confirm exact match and keep together; consider excluding one working copy"
                        if match_type == "exact"
                        else "compare contact sheet; confirm as one group or reject false positive"
                        if group_id
                        else "none"
                    ),
                }
            )
    summary_groups = []
    for group_id, members in sorted(
        (
            (group_for_path[component[0]], component)
            for component in duplicate_components
        ),
        key=lambda item: item[0],
    ):
        distances = [
            hamming_distance(
                by_path[first]["perceptual_hash"], by_path[second]["perceptual_hash"]
            )
            for index, first in enumerate(members)
            for second in members[index + 1 :]
        ]
        exact = any(
            tuple(sorted((first, second))) in exact_pairs
            for index, first in enumerate(members)
            for second in members[index + 1 :]
        )
        summary_groups.append(
            {
                "group_id": group_id,
                "image_paths": members,
                "match_type": "exact" if exact else "near",
                "minimum_hamming_distance": min(distances) if distances else None,
                "review_status": "pending",
                "recommended_review_action": (
                    "confirm exact match and keep together; consider excluding one working copy"
                    if exact
                    else "compare contact sheet; confirm as one group or reject false positive"
                ),
            }
        )
    summary = {
        "included_images_analyzed": len(records),
        "dhash_bits": 64,
        "hamming_threshold": threshold,
        "exact_duplicate_groups": sum(group["match_type"] == "exact" for group in summary_groups),
        "near_duplicate_candidate_groups": sum(
            group["match_type"] == "near" for group in summary_groups
        ),
        "candidate_images": sum(len(group["image_paths"]) for group in summary_groups),
        "pending_groups": len(summary_groups),
        "groups": summary_groups,
    }
    output_path.with_name("duplicate_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(duplicate_components)
