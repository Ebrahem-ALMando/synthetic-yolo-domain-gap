# Project State

Last updated: 2026-07-18

## Current sprint

Sprint 2 — Candidate Dataset Validation, Acquisition, Audit, and Immutable Real-Data Split.

## Completed

- Preserved the Sprint 1 foundation, deterministic seed 42, protocols, and ignore policy.
- Approved the CC BY 4.0 Aquarium Combined version-2 `raw-1024` export after local verification.
- Verified the recorded archive SHA-256 and confirmed all 1,281 raw stored files are read-only.
- Read the stable seven-class numeric order from the acquired `data.yaml`.
- Validated all 638 images and 638 matching labels without modifying raw data.
- Accepted 635 images with 4,784 valid objects and recorded 3 strict exclusions.
- Generated machine-readable validation/audit data, a Markdown audit, and eight real-data plots.
- Found no exact accepted-image duplicates and generated 8 pending dHash groups plus contact sheets.
- Proposed 83 conservative source groups and generated 58 source contact-sheet pages.
- Confirmed 17 explicit `_MOV` groups covering 81 frames from filename evidence.
- Strengthened split creation to reject incomplete duplicate and source review statuses.

## Human review blocker

- All 8 dHash duplicate candidate groups remain `pending`.
- Of 83 source groups, 17 are confirmed and 66 remain `pending`; the pending groups cover 554 still
  images.
- Some consecutive-number contact sheets span different exhibits, so automatic confirmation would
  create unsupported provenance claims.
- No train, validation, or test manifest has been created. No split is frozen and no manifest hash,
  combined split identity, or real leakage result exists.

## Data and results status

The approved raw dataset is local and ignored by Git. Generated audit and review outputs are also
ignored. There are no synthetic images, models, training runs, inference results, fabricated metrics,
or synthetic source/background manifests.

## Next gate

Complete the instructions in `reports/dataset_audit/aquarium/review_instructions.md`. After every
duplicate and source group has an accepted final status and stable grouping, rerun split creation,
then the hard-fail leakage checker. Sprint 3 cannot begin before those checks pass.
