# Dataset Protocol

## Status

Aquarium Combined version 2 (`raw-1024`) is approved after local file validation and complete visual
review. The accepted set is 635 of 638 acquired images; three exclusions are recorded in the
machine-readable audit. Class order comes from the acquired `data.yaml`. Seed-42 manifests are now
frozen under `manifests/aquarium`. See `docs/datasets/aquarium_candidate.md` and the datasheet.

## Candidate validation

Record the source, version, license, annotation format, class distribution, image provenance,
duplicate rate, corrupt files, object-size distribution, and suitability for synthetic generation.
Dataset acquisition must be scripted or documented with checksums where licensing permits.

The raw ZIP and extracted export live under `datasets/raw/aquarium`, are recorded with a SHA-256
checksum, and are treated as immutable. Validation and normalization write only to ignored working
and report directories. The validator does not silently repair annotations.

## Split order and isolation

1. Inventory and validate all candidate real images and annotations.
2. Detect exact and near duplicates before splitting.
3. Group related frames, scenes, sources, or sequences so they cannot cross splits.
4. Create train, validation, and held-out test manifests with deterministic seed 42.
5. Freeze the real test manifest before synthetic generation or model development.
6. Record dataset version, split procedure, counts, hashes, and any exclusions.

The target ratios are 70% train, 20% validation, and 10% test. Allocation operates on reviewed
source/duplicate groups and uses image-level multi-label classes in a deterministic greedy balancing
procedure. Exact requested percentages may be impossible for indivisible groups; actual counts and
trade-offs must be recorded.

The fixed real test set must never change between experiments. Its images and annotations must never
be used as training or validation samples, copy-paste object sources, synthetic backgrounds,
augmentation inputs, or references that influence synthetic generation. Validation remains separate
from the test set and is used for training-time decisions.

## Leakage checks

Before generating or training, automatically verify that file identities, content hashes, and known
source groups do not overlap between the test manifest and any training, validation, source-object,
or background manifest. A detected overlap is a hard failure, not a warning.

Exact duplicates use SHA-256 of image bytes. Near-duplicate candidates use a 64-bit difference hash
(dHash): resize grayscale pixels to 9x8, compare adjacent horizontal values, and group hashes whose
Hamming distance is at most the configured threshold (default 6). Candidates require human review;
files are never automatically deleted. Split creation rejects any duplicate group whose review
status is still `pending`. Source grouping likewise requires `review_status` of `confirmed` or
`not_applicable`; `pending`, `split_required`, and `merge_required` block the split. Review integrity
also requires every accepted image exactly once, every excluded image zero times, non-empty stable
group identities, existing referenced paths, and compatible source assignments for confirmed
duplicate pairs.

The frozen Aquarium split has 444/128/63 train/validation/test images. Group integrity is stronger
than exact class coverage: all `penguin` examples belong to one repeated-exhibit group and therefore
remain train-only. The fixed test set contains all other six classes and may not be revised to improve
this distribution.

The project object-size rule uses bounding-box pixel area in the inspected image: small is below
32^2 pixels, medium is from 32^2 up to but excluding 96^2, and large is at least 96^2.

## Repository policy

Raw data, processed images, labels, and generated synthetic data live under ignored `datasets/`
paths. Only small metadata, scripts, schemas, and legally permissible frozen manifests may be
versioned. Raw files and generated audit/review outputs remain ignored.
