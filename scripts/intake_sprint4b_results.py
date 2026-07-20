"""Validate and safely extract the returned Sprint 4B V2 CUDA results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.training.intake import (  # noqa: E402
    ARCHIVE_NAME,
    extract_validated_archive,
    validate_extracted_runs,
    validate_return_archive,
    write_intake_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/external_training/sprint4b-v2",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "reports/training"
    )
    parser.add_argument(
        "--reuse-extraction",
        action="store_true",
        help="Validate an existing extracted directory; never overwrite it.",
    )
    parser.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help="Skip loadability checks for diagnostics only; not valid for final intake.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact_dir = args.artifact_dir.resolve()
    extracted = artifact_dir / "extracted"
    try:
        archive_validation = validate_return_archive(artifact_dir)
        if extracted.exists():
            if not args.reuse_extraction:
                raise FileExistsError(
                    "Extraction already exists; pass --reuse-extraction to validate it "
                    "without overwrite"
                )
        else:
            extract_validated_archive(artifact_dir / ARCHIVE_NAME, extracted)
        validation = validate_extracted_runs(
            PROJECT_ROOT,
            artifact_dir,
            extracted,
            archive_validation,
            load_checkpoints=not args.skip_checkpoint_load,
        )
        reports = write_intake_reports(args.report_dir.resolve(), archive_validation, validation)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "archive_sha256": archive_validation["archive_sha256"],
                    "training_identity": validation["training_identity"],
                    "regimes": [run["regime"] for run in validation["runs"]],
                    "checkpoint_loadability_verified": validation[
                        "checkpoint_loadability_verified"
                    ],
                    "test_set_access_count": 0,
                    "reports": [path.relative_to(PROJECT_ROOT).as_posix() for path in reports],
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
