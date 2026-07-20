from __future__ import annotations

import hashlib
import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from synthdet_api.config import Settings
from synthdet_api.main import create_app


def _png(color: str = "blue") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeModelService:
    def __init__(self, original) -> None:
        self.original = original

    def validate_upload(self, *args, **kwargs):
        return self.original.validate_upload(*args, **kwargs)

    def infer(self, **kwargs):
        image = kwargs["image"]
        return {
            "model_id": kwargs["model_id"],
            "filename": kwargs["filename"],
            "original_width": image.width,
            "original_height": image.height,
            "detections": [],
            "detection_count": 0,
            "preprocessing_duration_ms": 1.0,
            "inference_duration_ms": 2.0,
            "postprocessing_duration_ms": 1.0,
            "total_duration_ms": 4.0,
            "device": "cpu",
            "annotated_image_mime": None,
            "annotated_image_base64": None,
        }


def _client(tmp_path: Path, max_bytes: int = 1024 * 1024) -> TestClient:
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        repository_root=root,
        model_root=tmp_path / "missing-models",
        max_upload_bytes=max_bytes,
    )
    app = create_app(settings)
    return TestClient(app)


def test_health_registry_and_openapi(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/health").json()["status"] == "ok"
    models = client.get("/api/v1/models").json()
    assert len(models["models"]) == 5
    assert models["recommended_model"] == "real_only"
    assert all(not model["available"] for model in models["models"])
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/inference" in paths
    assert "/api/v1/reproducibility" in paths


def test_invalid_mime_size_and_protected_hash_are_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path, max_bytes=128)
    invalid = client.post("/api/v1/inference", files={"file": ("bad.txt", b"hello", "text/plain")})
    assert invalid.status_code == 415
    oversized = client.post(
        "/api/v1/inference", files={"file": ("large.png", b"x" * 129, "image/png")}
    )
    assert oversized.status_code == 413
    protected = _png("red")
    client.app.state.repository.__dict__["protected_hashes"] = frozenset(
        {hashlib.sha256(protected).hexdigest()}
    )
    blocked = client.post(
        "/api/v1/inference", files={"file": ("external.png", protected, "image/png")}
    )
    assert blocked.status_code == 403
    assert blocked.json()["code"] == "protected_test_image"


def test_inference_flow_uses_bounded_external_upload(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.app.state.model_service = FakeModelService(client.app.state.model_service)
    response = client.post(
        "/api/v1/inference",
        files={"file": ("demo.png", _png(), "image/png")},
        data={"model_id": "real_only", "confidence": "0.25", "iou": "0.7"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "real_only"
    assert body["original_width"] == 32
    assert body["detections"] == []


def test_missing_model_and_unknown_model_are_structured(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/api/v1/models/nope").status_code == 404
    response = client.post(
        "/api/v1/inference",
        files={"file": ("demo.png", _png(), "image/png")},
        data={"model_id": "real_only"},
    )
    assert response.status_code == 503
    assert response.json()["code"] == "model_unavailable"
