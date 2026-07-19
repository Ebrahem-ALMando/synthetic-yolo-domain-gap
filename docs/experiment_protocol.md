# Experiment Protocol

## Frozen five-regime design

| Regime | Real | Synthetic | Real percentage | Synthetic percentage |
| --- | ---: | ---: | ---: | ---: |
| `synthetic_only` | 0 | 427 | 0.000000% | 100.000000% |
| `real_25` | 107 | 320 | 25.058548% | 74.941452% |
| `real_50` | 214 | 213 | 50.117096% | 49.882904% |
| `real_75` | 320 | 107 | 74.941452% | 25.058548% |
| `real_only` | 427 | 0 | 100.000000% | 0.000000% |

The equal 427-image budget is fixed. Seed-42 deterministic multi-label deficit selection chooses the
real subset in each mixed regime. Synthetic samples are the exact base-canvas complement. No regime
can contain both a real image and its synthetic derivative, omit a canvas, or represent a canvas
twice. Class coverage is preserved in every constructed regime.

Manifest hashes and the combined identity are frozen in
`manifests/aquarium/experiments/v1/experiment_metadata.json`. Generated class/object/source-group
tables and the composition figure are reproducible with `python scripts/audit_experiments.py`.

## Controlled variables

Every regime uses the same real validation set, YOLO11n architecture, pretrained initialization,
image size, epochs, batch, optimizer, learning-rate schedule, augmentation, early stopping, seed,
and thresholds. Only real-versus-synthetic training-file composition changes.

## Protected test policy

The active real test manifest is read only for frozen identity and leakage protection in Sprint 4A.
Test images are not materialized, loaded by YOLO, inspected, predicted, trained on, validated on, or
used to tune any setting. Final test evaluation belongs to a later sprint after all five primary
training runs and their configuration are frozen.

## Traceability

Every run receives a unique ignored directory containing machine-written metadata, resolved
configuration, Ultralytics logs, and checkpoints. Smoke results are marked
`scientific_result: false`. Failed and interrupted attempts remain honestly labeled. Metrics must
never be copied into versioned reports by hand.
