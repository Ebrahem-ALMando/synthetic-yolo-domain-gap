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
python scripts/finalize_aquarium_review.py
python scripts/validate_reviews.py reports/dataset_audit/aquarium/image_records.csv reports/dataset_audit/aquarium/duplicate_candidates.csv reports/dataset_audit/aquarium/reviewed_source_groups.csv --dataset-root datasets/raw/aquarium/export
python scripts/create_real_splits.py reports/dataset_audit/aquarium/image_records.csv reports/dataset_audit/aquarium/duplicate_candidates.csv --source-groups reports/dataset_audit/aquarium/reviewed_source_groups.csv --dataset-root datasets/raw/aquarium/export --preview
python scripts/create_real_splits.py reports/dataset_audit/aquarium/image_records.csv reports/dataset_audit/aquarium/duplicate_candidates.csv --source-groups reports/dataset_audit/aquarium/reviewed_source_groups.csv --dataset-root datasets/raw/aquarium/export
python scripts/create_real_splits.py reports/dataset_audit/aquarium/image_records.csv reports/dataset_audit/aquarium/duplicate_candidates.csv --source-groups reports/dataset_audit/aquarium/reviewed_source_groups.csv --dataset-root datasets/raw/aquarium/export --verify-frozen
python scripts/check_leakage.py manifests/aquarium
python scripts/audit_real_split.py manifests/aquarium reports/dataset_audit/aquarium/bounding_boxes.csv
```

Automatic acquisition requires `ROBOFLOW_API_KEY`. For manual acquisition, download version 2
`raw-1024` as YOLOv5 PyTorch from the official Roboflow project, then run:

```bash
python scripts/acquire_aquarium.py --archive path/to/downloaded.zip
```

Raw and generated outputs are ignored. Frozen manifests contain only repository-relative public-data
paths and hashes and must not be overwritten. The split command refuses existing outputs.

The completed review plan is versioned at `configs/datasets/aquarium_review.yaml`. Generated review
logs remain under the ignored audit path. The split command rejects unresolved reviews, refuses to
overwrite frozen outputs, and can reproduce the split in temporary storage with `--verify-frozen`.
The old Roboflow export folder names remain path components only; their train/valid/test labels were
not used as scientific split assignments.

## Sprint 2.5 Penguin review and active split

```bash
python scripts/review_penguin_groups.py
python scripts/validate_reviews.py reports/dataset_audit/aquarium/image_records.csv reports/dataset_audit/aquarium/duplicate_candidates.csv reports/dataset_audit/aquarium/penguin_review/reviewed_source_groups_v2.csv --dataset-root datasets/raw/aquarium/export
python scripts/create_real_splits.py reports/dataset_audit/aquarium/image_records.csv reports/dataset_audit/aquarium/duplicate_candidates.csv --source-groups reports/dataset_audit/aquarium/penguin_review/reviewed_source_groups_v2.csv --dataset-root datasets/raw/aquarium/export --seed 42 --output manifests/aquarium/v2 --verify-frozen
python scripts/check_leakage.py manifests/aquarium/v2
python scripts/audit_real_split.py manifests/aquarium/v2 reports/dataset_audit/aquarium/bounding_boxes.csv --output reports/dataset_audit/aquarium/penguin_review/v2_split_audit
```

Split V1 hashes remain recorded in `manifests/aquarium/v1/split_metadata.json`; seed 42 reproduces
identity `c926fd840a05385e604682d647b57f2d496c5d31c96f02ad7f4b33eba29b7db4`. Active Split V2 hashes
are in `manifests/aquarium/v2/split_metadata.json`; seed 42 reproduces identity
`02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7`. Synthetic source and
background manifests were absent at the end of Sprint 2.5.

## Sprint 3 copy-paste pool

```bash
python scripts/build_object_bank.py
python scripts/generate_synthetic.py --smoke
python scripts/validate_synthetic.py --output datasets/processed/aquarium/synthetic/smoke --manifests reports/synthetic_audit/aquarium/smoke/manifests --count 16
python scripts/audit_synthetic.py --output datasets/processed/aquarium/synthetic/smoke --manifests reports/synthetic_audit/aquarium/smoke/manifests --report reports/synthetic_audit/aquarium/smoke
python scripts/generate_synthetic.py --full
python scripts/validate_synthetic.py
python scripts/check_leakage.py manifests/aquarium/v2 --expected-split-identity 02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7 --synthetic-source-manifest manifests/aquarium/synthetic/v1/synthetic_sources.csv --synthetic-background-manifest manifests/aquarium/synthetic/v1/synthetic_backgrounds.csv --synthetic-image-manifest manifests/aquarium/synthetic/v1/synthetic_images.csv
python scripts/audit_synthetic.py
python scripts/generate_synthetic.py --verify-frozen
```

The full pool uses root seed 42, configuration hash
`7b957f23b46c760e4df446a362a7e1e8f194a54827696880c39c4b905b180eef`, object-bank identity
`22d5de79528f5de87b19bae606a93c62af357fc90ad51bfb81e4d197919c54d3`, and combined pool
identity `3dbd84054e5b2f9d95a3841974cf9c8bd3b987dcd5b84da0be91a06d9b0989ec`.

Reproduction writes to temporary storage, compares manifest identity and every image/label hash, and
never overwrites frozen outputs. Timestamps are excluded from identity. JPEG byte reproduction is
environment-sensitive, so the recorded Pillow/OpenCV/NumPy versions are part of the reproducibility
record even though they are not generation inputs.
