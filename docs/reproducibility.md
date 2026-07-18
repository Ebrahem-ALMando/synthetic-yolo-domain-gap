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

## Foundation checks

```bash
python scripts/check_environment.py
pytest
ruff check .
```

## Sprint 2 data commands

The commands below are run from the repository root. They do not authorize synthetic generation or
training.

```bash
python scripts/acquire_aquarium.py --dry-run
python scripts/acquire_aquarium.py --download
python scripts/validate_dataset.py datasets/raw/aquarium/export
python scripts/audit_dataset.py reports/dataset_audit/aquarium
python scripts/analyze_duplicates.py reports/dataset_audit/aquarium/image_records.csv
python scripts/propose_source_groups.py reports/dataset_audit/aquarium/image_records.csv reports/dataset_audit/aquarium/duplicate_candidates.csv datasets/raw/aquarium/export
python scripts/create_real_splits.py reports/dataset_audit/aquarium/image_records.csv reports/dataset_audit/aquarium/duplicate_candidates.csv --source-groups reports/dataset_audit/aquarium/reviewed_source_groups.csv
python scripts/check_leakage.py manifests/aquarium
```

Automatic acquisition requires `ROBOFLOW_API_KEY`. For manual acquisition, download version 2
`raw-1024` as YOLOv5 PyTorch from the official Roboflow project, then run:

```bash
python scripts/acquire_aquarium.py --archive path/to/downloaded.zip
```

Raw and generated outputs are ignored. Frozen manifests contain only repository-relative public-data
paths and hashes and must not be overwritten. The split command refuses existing outputs.

Current real-data execution stops after source proposals. Review
`reports/dataset_audit/aquarium/review_instructions.md`, the duplicate sheets, and all applicable
source sheets. The split command rejects `pending`, `split_required`, or `merge_required` source
statuses and pending duplicate groups. The old Roboflow export split folders are not provenance and
must not be used to resolve review decisions.
