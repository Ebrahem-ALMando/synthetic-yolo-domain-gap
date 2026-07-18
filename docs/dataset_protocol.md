# Dataset Protocol

## Status

The dataset is undecided in Sprint 1. Aquarium Object Detection is only a candidate and must be
validated in Sprint 2 before its class names or paths are finalized.

## Candidate validation

Record the source, version, license, annotation format, class distribution, image provenance,
duplicate rate, corrupt files, object-size distribution, and suitability for synthetic generation.
Dataset acquisition must be scripted or documented with checksums where licensing permits.

## Split order and isolation

1. Inventory and validate all candidate real images and annotations.
2. Detect exact and near duplicates before splitting.
3. Group related frames, scenes, sources, or sequences so they cannot cross splits.
4. Create train, validation, and held-out test manifests with deterministic seed 42.
5. Freeze the real test manifest before synthetic generation or model development.
6. Record dataset version, split procedure, counts, hashes, and any exclusions.

The fixed real test set must never change between experiments. Its images and annotations must never
be used as training or validation samples, copy-paste object sources, synthetic backgrounds,
augmentation inputs, or references that influence synthetic generation. Validation remains separate
from the test set and is used for training-time decisions.

## Leakage checks

Before generating or training, automatically verify that file identities, content hashes, and known
source groups do not overlap between the test manifest and any training, validation, source-object,
or background manifest. A detected overlap is a hard failure, not a warning.

## Repository policy

Raw data, processed images, labels, and generated synthetic data live under ignored `datasets/`
paths. Only small metadata, scripts, schemas, and legally permissible manifests may be versioned.
No dataset content is included in Sprint 1.

