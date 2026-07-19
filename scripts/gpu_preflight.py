"""Run the bounded CUDA memory test and freeze one Sprint 4B batch profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthdet.synthetic.contracts import sha256_file  # noqa: E402
from synthdet.training.cuda import (  # noqa: E402
    freeze_profile,
    inspect_cuda,
    select_common_profile,
    write_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--representative-regime", default="real_50", choices=("real_50",))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        common = yaml.safe_load((PROJECT_ROOT / "configs/training/common.yaml").read_text())
        weight_contract = yaml.safe_load(
            (PROJECT_ROOT / "configs/training/base_weight.yaml").read_text()
        )
        weight = (PROJECT_ROOT / args.weight).resolve()
        if not weight.is_file() or sha256_file(weight) != weight_contract["sha256"]:
            raise ValueError("yolo11n.pt is missing or differs from the frozen SHA-256")
        environment = inspect_cuda(args.device, PROJECT_ROOT)
        data = (
            PROJECT_ROOT
            / f"datasets/experiments/aquarium/v1/{args.representative_regime}/data.yaml"
        )
        if not data.is_file():
            raise FileNotFoundError(f"Representative dataset view is missing: {data}")

        def attempt(profile_name: str, batch: int) -> dict[str, object]:
            import torch
            from ultralytics import YOLO

            torch.cuda.empty_cache()
            run_root = PROJECT_ROOT / "artifacts/preflight"
            model = YOLO(str(weight))
            try:
                model.train(
                    data=str(data),
                    epochs=1,
                    imgsz=640,
                    batch=batch,
                    device=args.device,
                    workers=2,
                    seed=42,
                    deterministic=True,
                    optimizer="AdamW",
                    lr0=0.001,
                    weight_decay=0.0005,
                    cos_lr=True,
                    warmup_epochs=3,
                    patience=0,
                    cache=False,
                    pretrained=True,
                    save=False,
                    plots=False,
                    val=False,
                    fliplr=0.5,
                    flipud=0.0,
                    hsv_h=0.015,
                    hsv_s=0.4,
                    hsv_v=0.3,
                    scale=0.25,
                    translate=0.10,
                    degrees=5.0,
                    mosaic=0.5,
                    close_mosaic=10,
                    copy_paste=0.0,
                    mixup=0.0,
                    project=str(run_root),
                    name=f"{profile_name}-memory-check",
                    exist_ok=False,
                    resume=False,
                )
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(f"CUDA out of memory at batch {batch}: {error}") from error
            torch.cuda.synchronize(int(args.device))
            free_after, _ = torch.cuda.mem_get_info(int(args.device))
            return {
                "profile_name": profile_name,
                "batch": batch,
                "imgsz": 640,
                "status": "passed",
                "free_vram_bytes_after": free_after,
                "scientific_result": False,
                "test_set_access_count": 0,
            }

        selected, attempts = select_common_profile(attempt, environment["vram_total_bytes"])
        profile = freeze_profile(selected, attempts, environment, common, weight, args.device)
        write_profile(profile, args.output.resolve())
        print(json.dumps(profile, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
