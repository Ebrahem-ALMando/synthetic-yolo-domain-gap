"""Validate a Sprint 4B archive or an already extracted training bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.synthetic.contracts import sha256_file  # noqa: E402
from synthdet.training.bundle import safe_extract, validate_extracted_bundle  # noqa: E402
from synthdet.training.colab import resolve_expected_revision  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle", type=Path)
    group.add_argument("--extracted-root", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--expected-revision",
        help="Optional assertion; the validated internal inventory is authoritative.",
    )
    parser.add_argument("--skip-experiment-validation", action="store_true")
    return parser


def _validate_root(
    root: Path, run_experiment_validation: bool, revision_override: str | None
) -> dict[str, object]:
    inventory = validate_extracted_bundle(root)
    revision = resolve_expected_revision(root, revision_override)
    if run_experiment_validation:
        subprocess.run(
            [sys.executable, "scripts/materialize_training_bundle.py"],
            cwd=root,
            check=True,
        )
        subprocess.run([sys.executable, "scripts/validate_experiments.py"], cwd=root, check=True)
    return {
        "bundle_identity": inventory["bundle_identity"],
        "expected_repository_revision": revision,
        "source_branch": inventory["source_branch"],
        "source_worktree_dirty": inventory["source_worktree_dirty"],
        "file_count": inventory["file_count"],
        "experiment_validation": run_experiment_validation,
        "test_images_present": False,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.extracted_root:
            result = _validate_root(
                args.extracted_root.resolve(),
                not args.skip_experiment_validation,
                args.expected_revision,
            )
        else:
            bundle = args.bundle.resolve()
            actual = sha256_file(bundle)
            expected = args.expected_sha256
            sidecar = bundle.with_suffix(bundle.suffix + ".sha256")
            if expected is None and sidecar.is_file():
                expected = sidecar.read_text(encoding="utf-8").split()[0]
            if expected is None or actual != expected:
                raise ValueError(f"Archive SHA-256 mismatch: expected {expected}, got {actual}")
            with tempfile.TemporaryDirectory(prefix="synthdet-bundle-validation-") as temporary:
                root = Path(temporary) / "extracted"
                safe_extract(bundle, root)
                result = _validate_root(
                    root, not args.skip_experiment_validation, args.expected_revision
                )
                result["archive_sha256"] = actual
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
