"""Data-derived audit tables for the controlled experiment design."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from synthdet.synthetic.contracts import write_csv


def _label_counts(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
    return counts


def audit_experiment_design(
    regimes: dict[str, list[dict[str, str]]],
    metadata: dict[str, Any],
    validation_rows: list[dict[str, str]],
    class_names: list[str],
    project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    regime_summary: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for regime, rows in regimes.items():
        real = sum(row["sample_type"] == "real" for row in rows)
        unique_canvases = len({row["underlying_real_canvas_path"] for row in rows})
        regime_summary.append(
            {
                "regime": regime,
                "total": len(rows),
                "real": real,
                "synthetic": len(rows) - real,
                "real_percentage": f"{100 * real / len(rows):.6f}",
                "synthetic_percentage": f"{100 * (len(rows) - real) / len(rows):.6f}",
                "unique_underlying_canvases": unique_canvases,
                "complementary_pairing_passed": unique_canvases == len(rows),
            }
        )
        image_counts: Counter[int] = Counter()
        object_counts: Counter[int] = Counter()
        groups: Counter[str] = Counter(row["source_group_id"] for row in rows)
        for row in rows:
            ids = {int(value) for value in row["class_ids_present"].split(";") if value}
            image_counts.update(ids)
            object_counts.update(_label_counts(project_root / row["label_path"]))
        for class_id, class_name in enumerate(class_names):
            class_rows.append(
                {
                    "regime": regime,
                    "class_id": class_id,
                    "class_name": class_name,
                    "image_count": image_counts[class_id],
                    "object_count": object_counts[class_id],
                }
            )
        source_rows.extend(
            {"regime": regime, "source_group_id": group, "sample_count": count}
            for group, count in sorted(groups.items())
        )
    write_csv(output_dir / "regime_summary.csv", list(regime_summary[0]), regime_summary)
    write_csv(output_dir / "class_counts.csv", list(class_rows[0]), class_rows)
    write_csv(output_dir / "source_group_representation.csv", list(source_rows[0]), source_rows)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    names = [row["regime"] for row in regime_summary]
    real_counts = [int(row["real"]) for row in regime_summary]
    synthetic_counts = [int(row["synthetic"]) for row in regime_summary]
    axis.bar(names, real_counts, label="real")
    axis.bar(names, synthetic_counts, bottom=real_counts, label="synthetic")
    axis.set_ylabel("Training images")
    axis.set_title("Controlled 427-image training budget")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "regime_composition.png", dpi=160)
    plt.close(figure)
    summary = {
        "regimes": regime_summary,
        "validation_count": len(validation_rows),
        "validation_split_identity": metadata["real_split_identity"],
        "test_set_used": False,
        "regime_manifest_hashes": metadata["regime_manifest_hashes"],
        "combined_experiment_design_identity": metadata[
            "combined_experiment_design_identity"
        ],
    }
    (output_dir / "experiment_design_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
