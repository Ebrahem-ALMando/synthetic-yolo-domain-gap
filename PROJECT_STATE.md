# Project State

Last updated: 2026-07-20

Sprint 6A dashboard status: `sprint6a_dashboard_foundation_implemented`. An isolated Arabic-first
Next.js application now exists under `apps/web` with strict repository metadata, all planned
routes, RTL light/dark design, locked scientific results, labelled demo fixtures, snapshot
export/validation, tests, and a production build. The official brand component targets
`public/brand/synthdet-logo.png`; the supplied 2024 x 2024 bitmap is preserved unchanged there.
Sprint 6 is not complete; API,
result-archive, model-registry, and real inference integrations remain Sprint 6B.

Sprint 4B handoff status: `replacement_bundle_v2_required`. The notebook resolves the expected
revision from generated bundle inventory instead of a self-referential committed hash. The prior
dirty-source bundle and the runtime-incomplete v1 bundle are invalid and must not be uploaded.
Sprint 4B remains incomplete.

The subsequent v1 Colab attempt passed CUDA/profile preflight but exposed a missing frozen synthetic
`data.yaml` during the first final runner validation. Bundle v2 includes that train-only identity
descriptor, uses portable regime YAMLs, and requires all five final runner dry-runs during extracted
bundle validation. Sprint 4B remains incomplete pending the replacement bundle and external runs.

## Current sprint

Sprint 4A — Controlled Experiment Construction, Hardware Validation, and YOLO Smoke Training —
completed.

## Frozen input contract

- Active Aquarium Split V2: 427 train, 140 validation, 68 test.
- Real-split identity:
  `02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7`.
- Generator configuration:
  `7b957f23b46c760e4df446a362a7e1e8f194a54827696880c39c4b905b180eef`.
- Object bank: `22d5de79528f5de87b19bae606a93c62af357fc90ad51bfb81e4d197919c54d3`.
- Synthetic pool: `3dbd84054e5b2f9d95a3841974cf9c8bd3b987dcd5b84da0be91a06d9b0989ec`.
- Split V1 and all Sprint 3 provenance remain preserved. Frozen input, hash, and leakage checks pass.

## Sprint 4A completed

- Froze five 427-image regimes with exact real/synthetic counts 0/427, 107/320, 214/213, 320/107,
  and 427/0. Realized real percentages are 0, 25.058548, 50.117096, 74.941452, and 100.
- Validated complementary pairing: every underlying train canvas appears exactly once per regime,
  with no real image beside its synthetic derivative.
- Materialized ignored full 427/140 train/validation views and deterministic 16/14 smoke views for
  every regime. Class order, pairs, and annotations pass.
- Froze experiment-design identity
  `abe47eebc6567de98401e49e75279935cdeb0738558a40ee58dd2b423214ee4c` and reproduced every
  regime manifest hash in temporary storage.
- Recorded Windows 11, i7-1255U, 16.9 GB RAM, MX550 2 GB, Python 3.11.9, PyTorch 2.13.0 CPU, and
  Ultralytics 8.4.101. PyTorch reports no CUDA; classification is `smoke_training_only_cpu`.
- Completed all five one-epoch CPU smoke runs at 320 pixels, batch 2, on deterministic 16/14
  subsets. Checkpoints, logs, resolved settings, timings, status, and provenance are ignored and
  machine-recorded. Smoke metrics are not scientific results.
- Added safe dry-run/final runners, local and Colab execution paths, and a secret-free bundle
  builder. No full final run was started.

## Scientific limitations

The synthetic records remain copy-paste composites dominated by real-train pixels, not rendered
imagery. Multi-label selection preserves class coverage but cannot make all class/object frequencies
identical across regimes. Smoke subsets and one epoch establish pipeline operation only. This
machine's CPU run times do not predict final GPU run times or scientific performance.

## Data and results status

Generated datasets, caches, pretrained/smoke weights, smoke outputs, and audits remain ignored.
Versioned manifests, configuration, environment evidence, and documentation are protocol artifacts.
No full final experiment, protected-test inference, final evaluation, or operational API exists.
The Sprint 6A dashboard foundation displays repository metadata and explicit pending/demo states.

## Next gate

Build and validate bundle v2 from a clean committed `main` worktree so its inventory records that
clean HEAD and `source_worktree_dirty: false`. Only then run the versioned Colab notebook on CUDA.
Do not mark Sprint 4B complete until all five actual runs validate locally. The real test set remains
prohibited until the later fixed evaluation sprint.
