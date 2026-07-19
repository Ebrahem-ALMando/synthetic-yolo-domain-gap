"""Validate all five persistent runs and create the Sprint 4B return archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.training.completion import (  # noqa: E402
    build_completion_manifest,
    build_results_archive,
    write_validation_figure,
    write_validation_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        output = args.output_directory.resolve()
        output.mkdir(parents=True, exist_ok=True)
        completion = output / "training_completion_manifest.json"
        table = output / "NON_FINAL_VALIDATION_RESULTS.csv"
        figure = output / "NON_FINAL_VALIDATION_RESULTS.png"
        archive = output / "sprint4b_training_results.zip"
        manifest = build_completion_manifest(
            args.runs_root.resolve(),
            args.profile.resolve(),
            PROJECT_ROOT / "configs/training/common.yaml",
            PROJECT_ROOT / "manifests/aquarium/experiments/v1/experiment_metadata.json",
        )
        completion.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_validation_table(manifest, table)
        write_validation_figure(manifest, figure)
        exported = build_results_archive(manifest, archive, completion, table, figure)
        inventory_path = archive.with_suffix(archive.suffix + ".inventory.json")
        inventory_path.write_text(
            json.dumps(exported, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "completion_manifest": str(completion),
                    "training_identity": manifest["combined_sprint4b_training_identity"],
                    "results_archive": str(archive),
                    "archive_sha256": exported["archive_sha256"],
                    "test_set_access_count": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
