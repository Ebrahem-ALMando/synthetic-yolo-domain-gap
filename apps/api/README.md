# SynthDet FastAPI service

The API exposes sealed repository metadata and bounded, lazy YOLO inference. It never accepts server
filesystem paths, never returns private absolute paths, and rejects uploads whose SHA-256 belongs to
the protected 68-image Split V2 test set.

## Run locally

From the repository root, with the project virtual environment activated:

```powershell
$env:PYTHONPATH="src;apps/api"
python -m uvicorn synthdet_api.main:app --host 127.0.0.1 --port 8000
```

Linux/macOS:

```bash
PYTHONPATH=src:apps/api python -m uvicorn synthdet_api.main:app --host 127.0.0.1 --port 8000
```

The verified returned checkpoints are discovered automatically inside the ignored artifact tree.
For a portable demo directory, set `SYNTHDET_MODEL_ROOT`; it must contain
`<model_id>/best.pt` for each desired model. Every file is checked against the frozen SHA-256 before
loading. See `.env.example` for upload, timeout, device, and CORS controls.

OpenAPI is available at `/docs` and `/openapi.json`. The recommended model is `real_only`; models
are loaded one at a time and CPU is used automatically when CUDA is unavailable.

## Tests

```powershell
$env:PYTHONPATH="src;apps/api"
python -m pytest tests/test_api.py -q
```
