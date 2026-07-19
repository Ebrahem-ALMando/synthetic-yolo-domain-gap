"""Generate, freeze, or reproduce the Aquarium copy-paste synthetic pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.synthetic.contracts import load_synthetic_config, verify_active_split  # noqa: E402
from synthdet.synthetic.generator import (  # noqa: E402
    generate_synthetic_pool,
    verify_synthetic_reproduction,
)
from synthdet.synthetic.validation import validate_object_bank  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic/aquarium_synthetic_v1.yaml"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Generate the 16-image smoke gate.")
    mode.add_argument("--full", action="store_true", help="Freeze the 427-image primary pool.")
    mode.add_argument(
        "--verify-frozen",
        action="store_true",
        help="Regenerate into temporary storage and compare the frozen pool identity.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifests", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_synthetic_config(PROJECT_ROOT / args.config)
        split_dir = PROJECT_ROOT / Path(config.dataset["active_split_directory"])
        verify_active_split(split_dir, config.split_identity)
        logical_full = Path(config.dataset["output_directory"])
        full_output = PROJECT_ROOT / logical_full
        frozen_manifests = PROJECT_ROOT / Path(config.dataset["manifest_directory"])
        bank_errors = validate_object_bank(config, PROJECT_ROOT, full_output, frozen_manifests)
        if bank_errors:
            raise ValueError("Object-bank validation failed:\n- " + "\n- ".join(bank_errors))
        if args.verify_frozen:
            metadata = verify_synthetic_reproduction(
                config, PROJECT_ROOT, full_output, frozen_manifests
            )
        else:
            if args.smoke:
                logical_output = Path(args.output or "datasets/processed/aquarium/synthetic/smoke")
                manifest_path = Path(
                    args.manifests or "reports/synthetic_audit/aquarium/smoke/manifests"
                )
                count = int(config.dataset["smoke_pool_size"])
            else:
                logical_output = Path(args.output or logical_full)
                manifest_path = Path(args.manifests or config.dataset["manifest_directory"])
                count = int(config.dataset["full_pool_size"])
            metadata = generate_synthetic_pool(
                config,
                PROJECT_ROOT,
                PROJECT_ROOT / logical_output,
                logical_output,
                PROJECT_ROOT / manifest_path,
                frozen_manifests,
                full_output,
                count,
                ensure_class_coverage=args.smoke,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Synthetic generation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
