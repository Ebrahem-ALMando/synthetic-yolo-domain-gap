"""Read-only adapter over sealed tracked project metadata and local checkpoints."""

from __future__ import annotations

import csv
import json
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from synthdet.synthetic.contracts import sha256_file
from synthdet_api.config import Settings

REGIMES = ("synthetic_only", "real_25", "real_50", "real_75", "real_only")
ARABIC_NAMES = {
    "synthetic_only": "اصطناعي فقط",
    "real_25": "٢٥٪ بيانات حقيقية",
    "real_50": "٥٠٪ بيانات حقيقية",
    "real_75": "٧٥٪ بيانات حقيقية",
    "real_only": "حقيقي فقط",
}
COMPOSITIONS = {
    "synthetic_only": {"real": 0, "synthetic": 427},
    "real_25": {"real": 107, "synthetic": 320},
    "real_50": {"real": 214, "synthetic": 213},
    "real_75": {"real": 320, "synthetic": 107},
    "real_only": {"real": 427, "synthetic": 0},
}
CLASS_NAMES_AR = (
    "سمكة",
    "قنديل البحر",
    "بطريق",
    "ببغاء البحر",
    "قرش",
    "نجم البحر",
    "شفنين",
)


class ProjectRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.repository_root.resolve()

    def _json(self, relative: str) -> Any:
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    @cached_property
    def contract(self) -> dict[str, Any]:
        return yaml.safe_load(
            (self.root / "configs/evaluation/sprint5_final.yaml").read_text(encoding="utf-8")
        )

    @cached_property
    def final_results(self) -> dict[str, dict[str, Any]]:
        document = self._json("reports/evaluation/final_test_metrics.json")
        return {result["regime"]: result for result in document["models"]}

    @cached_property
    def campaign_lock(self) -> dict[str, Any]:
        return self._json("reports/evaluation/sprint5_campaign_lock.json")

    @cached_property
    def finalization_state(self) -> dict[str, Any]:
        return self._json("FINALIZATION_STATE.json")

    @cached_property
    def protected_hashes(self) -> frozenset[str]:
        path = self.root / self.contract["protected_test"]["manifest_path"]
        with path.open(encoding="utf-8", newline="") as handle:
            return frozenset(row["content_hash"] for row in csv.DictReader(handle))

    def checkpoint_path(self, model_id: str) -> Path:
        if self.settings.model_root is not None:
            return (self.settings.model_root / model_id / "best.pt").resolve()
        return (self.root / self.contract["checkpoints"][model_id]["path"]).resolve()

    def model_record(self, model_id: str) -> dict[str, Any]:
        if model_id not in REGIMES:
            raise KeyError(model_id)
        expected_hash = self.contract["checkpoints"][model_id]["sha256"]
        checkpoint = self.checkpoint_path(model_id)
        available = checkpoint.is_file()
        hash_matches = available and sha256_file(checkpoint) == expected_hash
        result = self.final_results[model_id]
        return {
            "model_id": model_id,
            "display_name_ar": ARABIC_NAMES[model_id],
            "checkpoint_path": f"models/{model_id}/best.pt",
            "checkpoint_sha256": expected_hash,
            "available": bool(available and hash_matches),
            "availability_reason": (
                "ready"
                if available and hash_matches
                else "hash_mismatch"
                if available
                else "checkpoint_missing"
            ),
            "recommended": model_id == "real_only",
            "training_composition": COMPOSITIONS[model_id],
            "class_names": self.contract["class_names"],
            "class_names_ar": list(CLASS_NAMES_AR),
            "checkpoint_size_bytes": result["checkpoint"]["size_bytes"],
            "parameter_count": result["checkpoint"]["parameter_count"],
            "flops_giga": result["checkpoint"]["flops_giga"],
            "final_metrics": result["metrics"],
        }

    def models(self) -> list[dict[str, Any]]:
        return [self.model_record(model_id) for model_id in REGIMES]

    def evaluation(self, model_id: str | None = None) -> Any:
        if model_id is not None:
            if model_id not in REGIMES:
                raise KeyError(model_id)
            result = self.final_results[model_id]
            return {
                "regime": model_id,
                "metrics": result["metrics"],
                "per_class": result["per_class"],
                "object_size_metrics": result["object_size_metrics"],
                "latency_ms_per_image": result["latency_ms_per_image"],
                "throughput_images_per_second": result["throughput_images_per_second"],
                "prediction_sha256": result["prediction_sha256"],
                "result_sha256": result["result_sha256"],
            }
        return {
            "campaign_id": self.campaign_lock["campaign_id"],
            "attempt_id": self.campaign_lock["attempt_id"],
            "status": self.campaign_lock["status"],
            "recommended_model": self.campaign_lock["recommended_model"],
            "ranking": self._csv("reports/evaluation/final_model_ranking.csv"),
            "models": [self.evaluation(regime) for regime in REGIMES],
        }

    def _csv(self, relative: str) -> list[dict[str, str]]:
        with (self.root / relative).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def training(self) -> dict[str, Any]:
        return {
            "status": "complete_verified",
            "training_identity": self.contract["training_identity"],
            "profile": {"name": "standard", "image_size": 640, "batch": 16, "gpu": "Tesla T4"},
            "test_access_count": 0,
            "regimes": self._csv("reports/training/sprint4b_v2_validation_summary.csv"),
        }

    def project(self) -> dict[str, Any]:
        return {
            "product": "سينث دِت — SynthDet",
            "project_status": self.finalization_state["project_status"],
            "phase": "post_evaluation_product_integration",
            "classes": self.contract["class_names"],
            "models_completed": 5,
            "recommended_model": "real_only",
            "primary_metric": "map50_95",
            "primary_metric_value": self.final_results["real_only"]["metrics"]["map50_95"],
            "campaign_id": self.campaign_lock["campaign_id"],
        }

    def reproducibility(self) -> dict[str, Any]:
        return {
            "repository_revision": self.campaign_lock["execution_revision"],
            "contract_commit": self.campaign_lock["contract_commit"],
            "contract_sha256": self.campaign_lock["contract_sha256"],
            "split_identity": self.contract["protected_test"]["real_split_identity"],
            "test_manifest_sha256": self.contract["protected_test"]["manifest_sha256"],
            "training_identity": self.contract["training_identity"],
            "campaign_id": self.campaign_lock["campaign_id"],
            "successful_campaign_count": 1,
            "failed_technical_attempt_count": len(self.campaign_lock.get("attempt_history", [])),
            "checkpoint_hashes": {
                model_id: self.contract["checkpoints"][model_id]["sha256"] for model_id in REGIMES
            },
        }

    def reports(self) -> list[dict[str, str]]:
        records = []
        for relative in (
            "reports/evaluation/sprint5_final_report.md",
            "reports/evaluation/final_model_ranking.md",
            "reports/analysis/error_analysis.md",
            "reports/training/sprint4b_v2_intake_report.md",
            "docs/evaluation_protocol.md",
        ):
            path = self.root / relative
            if path.is_file():
                records.append(
                    {
                        "id": relative.replace("/", "__"),
                        "title": path.stem.replace("_", " "),
                        "repository_path": relative,
                        "sha256": sha256_file(path),
                        "media_type": "text/markdown",
                    }
                )
        return records
