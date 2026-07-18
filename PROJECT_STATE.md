# Project State

Last updated: 2026-07-18

## Current sprint

Sprint 2 — Candidate Dataset Validation, Acquisition, Audit, and Immutable Real-Data Split —
completed.

## Completed

- Approved the CC BY 4.0 Aquarium Combined version-2 `raw-1024` export after local verification.
- Verified the archive SHA-256 and all 1,281 raw stored files as read-only.
- Validated all 638 image/label pairs without modifying raw data; accepted 635 images with 4,784
  objects and recorded 3 strict exclusions.
- Confirmed all 8 dHash duplicate groups after agent-assisted inspection of every duplicate sheet.
- Reviewed all 58 source contact-sheet pages and 22 pending singleton images. Seven original groups
  were split, 42 participated in visual merges, 29 remained unchanged, and 6 isolated images were
  approved as `not_applicable` singletons.
- Resolved all 635 accepted images into 52 stable source groups with zero pending, split-required, or
  merge-required states.
- Froze deterministic seed-42 manifests: 444 train (69.92%), 128 validation (20.16%), and 63 test
  (9.92%). The three excluded records remain outside every split.
- Reproduced the manifests non-destructively with combined identity
  `c926fd840a05385e604682d647b57f2d496c5d31c96f02ad7f4b33eba29b7db4`.
- Passed path, content-hash, duplicate-group, source-group, frozen-hash, and synthetic-manifest-aware
  leakage checks. No synthetic source/background manifests exist yet.
- Generated data-derived split tables and six split figures under the ignored dataset audit path.

## Frozen manifest hashes

- `real_train.csv`: `be8e7db0310612231cf1e63372c0eb7fb095c4a83f87826a131611e160049497`
- `real_val.csv`: `cf7fe91cc29c5708c2ce5b2085d48cf968af79da0ab420231e6ed73d8ec84cd2`
- `real_test.csv`: `6a928386222c2ee8e7b6a7e61ea977aa9ef3ed00fe580f4636935d0c2a9d7a8c`
- `excluded.csv`: `a308b2a48d5412b735934bfa42372d6b30acee35edf2823eb08efb5e0e0bdc55`
- `duplicate_groups.csv`: `32ddcf36f992e28df7b7b1cdd0fa99d5c3e00d9bf24d6ab0ea2e011b7c129348`

## Scientific limitation

All 71 `penguin`-class images share one conservatively reviewed repeated penguin-exhibit source
group. Group integrity therefore keeps that class train-only; validation and test retain all other
six classes. This limitation is preferred to background/source leakage.

## Data and results status

Raw data and generated audit/review outputs remain ignored. The small frozen manifests are protocol
artifacts. No synthetic images, model weights, training runs, inference results, API, dashboard, or
evaluation metrics exist.

## Next gate

Sprint 3 may begin from the fixed manifests only. The real test manifest must never change and its
images may never be training/validation samples, copy-paste sources, synthetic backgrounds,
augmentation inputs, or synthetic-generation references.
