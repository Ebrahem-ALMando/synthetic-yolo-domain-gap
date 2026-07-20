"""SynthDet FastAPI application factory."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from synthdet_api.config import Settings
from synthdet_api.repository import ProjectRepository
from synthdet_api.schemas import ErrorBody, HealthResponse, InferenceResponse
from synthdet_api.service import APIError, ModelService

LOGGER = logging.getLogger("synthdet.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    repository = ProjectRepository(settings)
    service = ModelService(settings, repository)
    application = FastAPI(
        title="SynthDet API",
        version="1.0.0",
        description=(
            "Read-only scientific metadata and bounded lazy inference for SynthDet. "
            "Protected Split V2 test images are rejected by SHA-256."
        ),
    )
    application.state.settings = settings
    application.state.repository = repository
    application.state.model_service = service
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.middleware("http")
    async def structured_request_log(request: Request, call_next):
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                json.dumps(
                    {
                        "event": "request_failed",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                    }
                )
            )
            raise
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            json.dumps(
                {
                    "event": "request_complete",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        )
        return response

    @application.exception_handler(APIError)
    async def api_error_handler(_request: Request, error: APIError) -> JSONResponse:
        body = ErrorBody(code=error.code, message=error.message, details=error.details)
        return JSONResponse(status_code=error.status_code, content=body.model_dump())

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="synthdet-api",
            project="SynthDet",
            evaluation_status=repository.campaign_lock["status"],
        )

    @application.get("/api/v1/project", tags=["project"])
    async def project() -> dict[str, Any]:
        return repository.project()

    @application.get("/api/v1/models", tags=["models"])
    async def models() -> dict[str, Any]:
        return {"models": repository.models(), "recommended_model": "real_only"}

    @application.get("/api/v1/models/{model_id}", tags=["models"])
    async def model(model_id: str) -> dict[str, Any]:
        try:
            return repository.model_record(model_id)
        except KeyError as error:
            raise APIError(404, "model_not_found", "Unknown model ID") from error

    @application.get("/api/v1/evaluation", tags=["science"])
    async def evaluation() -> dict[str, Any]:
        return repository.evaluation()

    @application.get("/api/v1/evaluation/{model_id}", tags=["science"])
    async def model_evaluation(model_id: str) -> dict[str, Any]:
        try:
            return repository.evaluation(model_id)
        except KeyError as error:
            raise APIError(404, "model_not_found", "Unknown model ID") from error

    @application.get("/api/v1/training", tags=["science"])
    async def training() -> dict[str, Any]:
        return repository.training()

    @application.get("/api/v1/reproducibility", tags=["science"])
    async def reproducibility() -> dict[str, Any]:
        return repository.reproducibility()

    @application.get("/api/v1/reports", tags=["reports"])
    async def reports() -> dict[str, Any]:
        return {"reports": repository.reports()}

    async def read_upload(upload: UploadFile) -> tuple[bytes, str, str]:
        data = await upload.read(settings.max_upload_bytes + 1)
        await upload.close()
        return data, upload.filename or "upload", upload.content_type or "application/octet-stream"

    async def infer_one(
        upload: UploadFile,
        model_id: str,
        confidence: float,
        iou: float,
        max_detections: int,
        annotate: bool,
    ) -> dict[str, Any]:
        data, filename, content_type = await read_upload(upload)
        active_service: ModelService = application.state.model_service
        image = active_service.validate_upload(data, filename, content_type)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    active_service.infer,
                    model_id=model_id,
                    image=image,
                    filename=filename,
                    confidence=confidence,
                    iou=iou,
                    max_detections=max_detections,
                    annotate=annotate,
                ),
                timeout=settings.inference_timeout_seconds,
            )
        except TimeoutError as error:
            raise APIError(
                504, "inference_timeout", "Inference exceeded the configured timeout"
            ) from error

    @application.post("/api/v1/inference", response_model=InferenceResponse, tags=["inference"])
    async def inference(
        file: Annotated[UploadFile, File(description="External JPEG, PNG, or WebP image")],
        model_id: Annotated[str, Form()] = "real_only",
        confidence: Annotated[float, Form(ge=0.001, le=1.0)] = 0.25,
        iou: Annotated[float, Form(ge=0.1, le=0.95)] = 0.7,
        max_detections: Annotated[int, Form(ge=1, le=300)] = 100,
        annotate: Annotated[bool, Form()] = True,
    ) -> dict[str, Any]:
        return await infer_one(file, model_id, confidence, iou, max_detections, annotate)

    @application.post("/api/v1/inference/batch", tags=["inference"])
    async def batch_inference(
        files: Annotated[
            list[UploadFile], File(description="At most four bounded external images")
        ],
        model_id: Annotated[str, Form()] = "real_only",
        confidence: Annotated[float, Form(ge=0.001, le=1.0)] = 0.25,
        iou: Annotated[float, Form(ge=0.1, le=0.95)] = 0.7,
    ) -> dict[str, Any]:
        if not files or len(files) > settings.max_batch_images:
            raise APIError(
                400,
                "invalid_batch_size",
                f"Batch size must be between 1 and {settings.max_batch_images}",
            )
        results = []
        for upload in files:
            results.append(await infer_one(upload, model_id, confidence, iou, 100, False))
        return {"model_id": model_id, "count": len(results), "results": results}

    return application


app = create_app()
