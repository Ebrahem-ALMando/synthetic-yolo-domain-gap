# Reproducibility

## Determinism

The project default seed is 42. Future entry points must seed Python, NumPy, augmentation libraries,
PyTorch, data loaders, and YOLO settings as applicable. Deterministic algorithms should be enabled
where practical, while known nondeterministic GPU behavior must be recorded rather than concealed.

## Environment

- Use Python 3.11 or newer and install from `pyproject.toml`.
- Record the OS, Python version, package versions, CUDA/runtime details, and accelerator model.
- Use repository-relative, `pathlib`-based paths for Windows and Linux portability.
- Do not encode local absolute paths, secrets, datasets, caches, weights, or runs in Git.

## Run identity

Every experiment configuration should fully describe its dataset manifests, mixture, training-image
budget, seed, model and training settings, synthetic generator settings, and evaluation thresholds.
Every output directory must identify that configuration and retain code revision and run metadata.

## Validation gates

Before a training run: validate annotations, frozen split identity, and absence of test leakage.
Before reporting: confirm run completion, evaluate from the recorded checkpoint on the fixed test
manifest, and generate tables/figures programmatically. Never fill missing results manually.

## Sprint 1 checks

```bash
python scripts/check_environment.py
pytest
ruff check .
```

