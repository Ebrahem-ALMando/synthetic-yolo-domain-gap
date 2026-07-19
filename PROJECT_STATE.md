# Project State

Last updated: 2026-07-19

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
No full experiment, test inference, final evaluation, API, or dashboard exists.

## Next gate

Sprint 4B may run the five frozen 50-epoch regimes on a CUDA-capable machine after selecting either
the standard batch-16 or predeclared low-memory batch-4 profile for all regimes. It must rerun all
identity, leakage, view, disk, and GPU checks. The real test set remains prohibited until the later
fixed evaluation sprint.
