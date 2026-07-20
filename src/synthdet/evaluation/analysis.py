"""Post-campaign validation and deterministic Sprint 5 error analysis."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from synthdet.evaluation.campaign import (
    REGIMES,
    _canonical_sha256,
    _iou_xywh,
    _size_name,
    _write_csv,
    _write_json,
    rank_models,
)
from synthdet.synthetic.contracts import sha256_file
from synthdet.training.intake import CLASS_NAMES


def _assert_finite(value: Any, label: str = "result") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {label}")


def validate_campaign_outputs(root: Path) -> dict[str, Any]:
    report_dir = root / "reports/evaluation"
    lock = json.loads((report_dir / "sprint5_campaign_lock.json").read_text(encoding="utf-8"))
    if (
        lock.get("status") != "completed_sealed"
        or lock.get("successful_campaign_count") != 1
        or lock.get("model_evaluation_passes") != 5
        or lock.get("completed_models") != list(REGIMES)
    ):
        raise ValueError("Sprint 5 lock is not one complete sealed five-model campaign")
    attempt = lock["attempt_id"]
    metrics_document = json.loads(
        (report_dir / "final_test_metrics.json").read_text(encoding="utf-8")
    )
    results = metrics_document.get("models", [])
    if [result.get("regime") for result in results] != list(REGIMES):
        raise ValueError("Final metrics do not contain the five regimes in frozen order")
    _assert_finite(results)
    hash_report = json.loads((report_dir / "sprint5_hash_report.json").read_text(encoding="utf-8"))
    for result in results:
        regime = result["regime"]
        raw_dir = root / f"artifacts/evaluation/{lock['campaign_id']}/{attempt}/{regime}"
        prediction_path = raw_dir / "per_image_predictions.json"
        ultralytics_path = raw_dir / "predictions.json"
        sealed_path = raw_dir / "sealed_result.json"
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        expected = hash_report["model_results"][regime]
        if sha256_file(prediction_path) != expected["prediction_sha256"]:
            raise ValueError(f"{regime} canonical prediction hash mismatch")
        if sha256_file(ultralytics_path) != expected["ultralytics_prediction_sha256"]:
            raise ValueError(f"{regime} Ultralytics prediction hash mismatch")
        hash_payload = dict(sealed)
        recorded_hash = hash_payload.pop("result_sha256")
        hash_payload.pop("result_hash_scope")
        if (
            _canonical_sha256(hash_payload) != recorded_hash
            or recorded_hash != expected["result_sha256"]
        ):
            raise ValueError(f"{regime} sealed result hash mismatch")
        per_image = json.loads(prediction_path.read_text(encoding="utf-8"))
        if len(per_image) != 68 or len({row["image_content_sha256"] for row in per_image}) != 68:
            raise ValueError(f"{regime} predictions do not cover 68 unique protected images")
        required = {
            "precision",
            "recall",
            "map50",
            "map50_95",
            "macro_per_class_ap50_95",
        }
        if not required.issubset(result["metrics"]):
            raise ValueError(f"{regime} metric schema is incomplete")
    expected_ranking = rank_models(results)
    with (report_dir / "final_model_ranking.csv").open(encoding="utf-8", newline="") as handle:
        ranking = list(csv.DictReader(handle))
    if [row["regime"] for row in ranking] != [row["regime"] for row in expected_ranking]:
        raise ValueError("Tracked ranking differs from the preregistered ranking function")
    figures = report_dir / "figures"
    required_figures = {
        "final_metric_comparison.png",
        "map50_95_vs_real_percentage.png",
        "map50_vs_real_percentage.png",
        "precision_recall_comparison.png",
        "per_class_ap_comparison.png",
        "object_size_comparison.png",
        "latency_comparison.png",
        "radar_comparison.png",
        "validation_vs_test_comparison.png",
        "synthetic_to_real_domain_gap_curve.png",
        "final_ranking.png",
    }
    missing = sorted(name for name in required_figures if not (figures / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing final evaluation figures: {missing}")
    return {
        "status": "valid",
        "campaign_id": lock["campaign_id"],
        "attempt_id": attempt,
        "models": 5,
        "protected_images_per_model": 68,
        "recommended_model": expected_ranking[0]["regime"],
        "failed_technical_attempts": len(lock.get("attempt_history", [])),
        "result_hashes_verified": 5,
        "prediction_hashes_verified": 10,
        "ranking_rule_verified": True,
    }


def _match_image(regime: str, image: dict[str, Any]) -> list[dict[str, Any]]:
    truths = image["ground_truth"]
    predictions = sorted(
        image["predictions"], key=lambda row: (-float(row["score"]), int(row["class_id"]))
    )
    matched_truth: set[int] = set()
    events: list[dict[str, Any]] = []
    base = {
        "regime": regime,
        "image_reference": image["image_reference"],
        "image_content_sha256": image["image_content_sha256"],
    }
    for prediction_index, prediction in enumerate(predictions):
        same_class = [
            (index, truth, _iou_xywh(prediction["bbox"], truth["bbox_xywh_pixels"]))
            for index, truth in enumerate(truths)
            if index not in matched_truth and truth["class_id"] == prediction["class_id"]
        ]
        best = max(same_class, key=lambda item: item[2], default=None)
        if best is not None and best[2] >= 0.5:
            matched_truth.add(best[0])
            events.append(
                {
                    **base,
                    "event_id": f"{regime}:{image['image_content_sha256']}:{prediction_index}:tp",
                    "error_type": "true_positive",
                    "pred_class_id": prediction["class_id"],
                    "pred_class_name": prediction["class_name"],
                    "gt_class_id": best[1]["class_id"],
                    "gt_class_name": best[1]["class_name"],
                    "confidence": float(prediction["score"]),
                    "iou": best[2],
                    "object_size": _size_name(best[1]["area_pixels"]),
                    "pred_bbox": prediction["bbox"],
                    "gt_bbox": best[1]["bbox_xywh_pixels"],
                }
            )
            continue
        all_overlaps = [
            (index, truth, _iou_xywh(prediction["bbox"], truth["bbox_xywh_pixels"]))
            for index, truth in enumerate(truths)
        ]
        overlap = max(all_overlaps, key=lambda item: item[2], default=None)
        same_overlap = max(same_class, key=lambda item: item[2], default=None)
        if (
            overlap is not None
            and overlap[2] >= 0.5
            and overlap[1]["class_id"] != prediction["class_id"]
        ):
            error_type = "class_confusion"
            target = overlap[1]
            iou = overlap[2]
        elif same_overlap is not None and same_overlap[2] >= 0.1:
            error_type = "localization_error"
            target = same_overlap[1]
            iou = same_overlap[2]
        else:
            error_type = "false_positive"
            target = overlap[1] if overlap else None
            iou = overlap[2] if overlap else 0.0
        events.append(
            {
                **base,
                "event_id": f"{regime}:{image['image_content_sha256']}:{prediction_index}:pred",
                "error_type": error_type,
                "pred_class_id": prediction["class_id"],
                "pred_class_name": prediction["class_name"],
                "gt_class_id": target["class_id"] if target else None,
                "gt_class_name": target["class_name"] if target else None,
                "confidence": float(prediction["score"]),
                "iou": iou,
                "object_size": _size_name(
                    float(prediction["bbox"][2]) * float(prediction["bbox"][3])
                ),
                "pred_bbox": prediction["bbox"],
                "gt_bbox": target["bbox_xywh_pixels"] if target else None,
            }
        )
    for truth_index, truth in enumerate(truths):
        if truth_index in matched_truth:
            continue
        nearest = max(
            (
                prediction
                for prediction in predictions
                if prediction["class_id"] == truth["class_id"]
            ),
            key=lambda prediction: _iou_xywh(prediction["bbox"], truth["bbox_xywh_pixels"]),
            default=None,
        )
        events.append(
            {
                **base,
                "event_id": f"{regime}:{image['image_content_sha256']}:{truth_index}:fn",
                "error_type": "false_negative",
                "pred_class_id": nearest["class_id"] if nearest else None,
                "pred_class_name": nearest["class_name"] if nearest else None,
                "gt_class_id": truth["class_id"],
                "gt_class_name": truth["class_name"],
                "confidence": float(nearest["score"]) if nearest else None,
                "iou": _iou_xywh(nearest["bbox"], truth["bbox_xywh_pixels"]) if nearest else 0.0,
                "object_size": _size_name(truth["area_pixels"]),
                "pred_bbox": nearest["bbox"] if nearest else None,
                "gt_bbox": truth["bbox_xywh_pixels"],
            }
        )
    return events


def _select_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for regime in REGIMES:
        model = [event for event in events if event["regime"] == regime]
        rules = (
            (
                "highest_confidence_false_positive",
                [event for event in model if event["error_type"] == "false_positive"],
                lambda event: (-(event["confidence"] or 0), event["image_content_sha256"]),
                8,
            ),
            (
                "highest_confidence_class_confusion",
                [event for event in model if event["error_type"] == "class_confusion"],
                lambda event: (-(event["confidence"] or 0), event["image_content_sha256"]),
                8,
            ),
            (
                "lowest_iou_matched_detection",
                [event for event in model if event["error_type"] == "true_positive"],
                lambda event: (event["iou"], -event["confidence"], event["image_content_sha256"]),
                8,
            ),
            (
                "small_object_miss_fixed_hash_order",
                [
                    event
                    for event in model
                    if event["error_type"] == "false_negative" and event["object_size"] == "small"
                ],
                lambda event: (event["image_content_sha256"], event["gt_class_id"]),
                8,
            ),
            (
                "largest_missed_object",
                [event for event in model if event["error_type"] == "false_negative"],
                lambda event: (
                    -(event["gt_bbox"][2] * event["gt_bbox"][3]),
                    event["image_content_sha256"],
                ),
                8,
            ),
        )
        for reason, candidates, key, count in rules:
            for event in sorted(candidates, key=key)[:count]:
                selected.append({**event, "selection_reason": reason})
        for class_id in range(len(CLASS_NAMES)):
            successes = [
                event
                for event in model
                if event["error_type"] == "true_positive" and event["gt_class_id"] == class_id
            ]
            if successes:
                event = sorted(
                    successes,
                    key=lambda row: (-row["iou"], -row["confidence"], row["image_content_sha256"]),
                )[0]
                selected.append({**event, "selection_reason": "representative_success_by_class"})
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for event in selected:
        unique[(event["event_id"], event["selection_reason"])] = event
    return list(unique.values())


def _gallery(root: Path, selected: list[dict[str, Any]]) -> int:
    gallery = root / "reports/analysis/error_gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    generated = 0
    for index, event in enumerate(selected):
        source = root / event["image_reference"]
        with Image.open(source) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        if event["gt_bbox"]:
            x, y, width, height = event["gt_bbox"]
            draw.rectangle((x, y, x + width, y + height), outline="#22c55e", width=4)
        if event["pred_bbox"]:
            x, y, width, height = event["pred_bbox"]
            draw.rectangle((x, y, x + width, y + height), outline="#ef4444", width=4)
        filename = (
            f"{index:04d}_{event['regime']}_{event['error_type']}_"
            f"{event['image_content_sha256'][:12]}.jpg"
        )
        canvas.save(gallery / filename, quality=88)
        event["gallery_filename"] = filename
        generated += 1
    return generated


def generate_error_analysis(root: Path) -> dict[str, Any]:
    validation = validate_campaign_outputs(root)
    lock = json.loads(
        (root / "reports/evaluation/sprint5_campaign_lock.json").read_text(encoding="utf-8")
    )
    events: list[dict[str, Any]] = []
    per_image_by_model: dict[str, list[dict[str, Any]]] = {}
    for regime in REGIMES:
        path = (
            root
            / "artifacts/evaluation"
            / lock["campaign_id"]
            / lock["attempt_id"]
            / regime
            / "per_image_predictions.json"
        )
        images = json.loads(path.read_text(encoding="utf-8"))
        per_image_by_model[regime] = images
        for image in images:
            events.extend(_match_image(regime, image))
    selected = _select_events(events)
    gallery_count = _gallery(root, selected)
    csv_rows = []
    for event in selected:
        row = dict(event)
        row["pred_bbox"] = json.dumps(row["pred_bbox"], separators=(",", ":"))
        row["gt_bbox"] = json.dumps(row["gt_bbox"], separators=(",", ":"))
        csv_rows.append(row)
    analysis_dir = root / "reports/analysis"
    _write_csv(analysis_dir / "error_cases.csv", csv_rows)
    confusions = Counter(
        (event["regime"], event["gt_class_name"], event["pred_class_name"])
        for event in events
        if event["error_type"] == "class_confusion"
    )
    confusion_rows = [
        {"regime": key[0], "ground_truth_class": key[1], "predicted_class": key[2], "count": count}
        for key, count in sorted(confusions.items(), key=lambda item: (-item[1], item[0]))
    ]
    _write_csv(
        analysis_dir / "class_confusions.csv",
        confusion_rows
        or [
            {"regime": "none", "ground_truth_class": "none", "predicted_class": "none", "count": 0}
        ],
    )
    image_stats: dict[str, dict[str, int]] = defaultdict(dict)
    for regime in REGIMES:
        for image in per_image_by_model[regime]:
            matched = sum(
                event["error_type"] == "true_positive"
                for event in events
                if event["regime"] == regime
                and event["image_content_sha256"] == image["image_content_sha256"]
            )
            image_stats[image["image_content_sha256"]][regime] = matched
    disagreements = []
    reference = {
        image["image_content_sha256"]: image["image_reference"]
        for image in per_image_by_model[REGIMES[0]]
    }
    for digest, counts in image_stats.items():
        disagreements.append(
            {
                "image_content_sha256": digest,
                "image_reference": reference[digest],
                **{f"{regime}_matched_detections": counts[regime] for regime in REGIMES},
                "matched_detection_range": max(counts.values()) - min(counts.values()),
            }
        )
    disagreements.sort(
        key=lambda row: (-row["matched_detection_range"], row["image_content_sha256"])
    )
    _write_csv(analysis_dir / "model_disagreement.csv", disagreements)
    counts = Counter((event["regime"], event["error_type"]) for event in events)
    summary = {
        "schema_version": 1,
        "campaign_id": lock["campaign_id"],
        "attempt_id": lock["attempt_id"],
        "analysis_only_not_used_for_ranking": True,
        "matching_iou": 0.5,
        "localization_band": "same-class IoU >=0.1 and <0.5",
        "confidence_floor": 0.001,
        "selection_is_deterministic": True,
        "event_counts": {
            regime: {error: counts[(regime, error)] for error in sorted({key[1] for key in counts})}
            for regime in REGIMES
        },
        "selected_metadata_rows": len(selected),
        "ignored_gallery_images_generated": gallery_count,
        "campaign_validation": validation,
    }
    _write_json(analysis_dir / "error_summary.json", summary)
    metrics = {
        row["regime"]: row
        for row in csv.DictReader(
            (root / "reports/evaluation/domain_gap_analysis.csv").open(encoding="utf-8", newline="")
        )
    }
    marginal = []
    for previous, current in zip(REGIMES[:-1], REGIMES[1:], strict=True):
        marginal.append(
            f"- `{previous}` → `{current}`: "
            f"{float(metrics[current]['map50_95']) - float(metrics[previous]['map50_95']):+.6f}."
        )
    report = [
        "# Sprint 5 Deterministic Error and Domain-Gap Analysis",
        "",
        "This analysis was generated only after the five-model campaign was sealed. It does not "
        "alter the preregistered ranking or evaluation thresholds.",
        "",
        "## Main finding",
        "",
        f"`real_only` ranks first at mAP@50-95 "
        f"{float(metrics['real_only']['map50_95']):.6f}; `synthetic_only` reaches "
        f"{float(metrics['synthetic_only']['map50_95']):.6f}. The absolute "
        f"gap is {float(metrics['real_only']['absolute_map50_95_change_from_synthetic_only']):.6f} "
        f"({float(metrics['real_only']['relative_map50_95_change_from_synthetic_only_percent']):.2f}%"
        " "
        "relative to synthetic-only).",
        "",
        "Every mixed regime exceeds synthetic-only, so the synthetic data supported non-zero test "
        "performance and the mixed sets improved on it. The curve is not monotonic: `real_50` is "
        "the strongest mixed regime, while `real_75` falls below both `real_25` and `real_50`. "
        "Consequently these results do not establish a causal or universally optimal ratio.",
        "",
        "## Sequential marginal changes in mAP@50-95",
        "",
        *marginal,
        "",
        "## Per-class and size findings",
        "",
        "The leading AP@50-95 regimes by class are: fish `real_50`; jellyfish `real_only`; "
        "penguin `synthetic_only`; puffin `real_25`; shark `real_only`; starfish `real_25`; and "
        "stingray `real_only`. Penguin is based on only four test images and must not be treated "
        "as stable evidence that synthetic-only generalizes better for that class.",
        "",
        "`real_only` leads the fixed small- and large-object strata; `real_50` leads medium "
        "objects. Small-object AP is weak for every regime, and the custom size-stratified AP is "
        "descriptive "
        "rather than a COCO area-range metric.",
        "",
        "## Error selection",
        "",
        "Selections use fixed ordering: highest-confidence false positives/confusions, lowest-IoU "
        "matches, hash-ordered small-object misses, largest missed objects, and one highest-IoU "
        "success per class and regime. `error_cases.csv` records every selection reason. Gallery "
        "bitmaps contain protected test pixels and remain ignored; only metadata is tracked.",
        "",
        "## Limitations",
        "",
        "The 68-image test split, class imbalance, four-image Penguin coverage, copy-paste "
        "synthetic "
        "generator, one architecture, and one dataset limit external validity. Differences are "
        "associational under this frozen design and are not uncertainty-adjusted causal effects.",
    ]
    (analysis_dir / "error_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary
