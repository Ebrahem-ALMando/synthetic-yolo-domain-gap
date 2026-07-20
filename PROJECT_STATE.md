# Project State

Last updated: 2026-07-20

Sprint 6A dashboard status: `sprint6a_dashboard_foundation_validated`. An isolated Arabic-first
Next.js application now exists under `apps/web` with strict repository metadata, all planned
routes, RTL light/dark design, locked scientific results, labelled demo fixtures, snapshot
export/validation, tests, and a production build. The official brand component targets
`public/brand/synthdet-logo.png`; the supplied 2024 x 2024 bitmap is preserved unchanged there.
The final Sprint 6A gate passed snapshot validation, strict TypeScript, ESLint, seven frontend unit
tests, production generation of all 15 Next.js routes, 54 Python tests, Ruff, and a headless-Chrome
audit of 13 primary routes at 390, 768, 1024, 1280, and 1440 pixels. That audit verifies HTTP
responses, RTL/language, unintended overflow, logo loading, mobile navigation, theme switching,
and browser console/page errors; its 65 ignored review screenshots were inspected on representative
mobile and desktop pages. Sprint 6 is not complete; API, verified-result, model-registry, and real
inference integrations remain Sprint 6B.

Sprint 4B status: `sprint4b_cuda_training_verified`. The returned v2 archive passed its external
SHA-256, internal/external inventory, all 68 per-file size/hash checks, safe extraction, forbidden
content checks, identity/configuration equality, 50-epoch CSV integrity, and loadability of all ten
best/last checkpoints. Exactly five final regimes completed on one frozen Tesla T4 `standard`
profile (640 pixels, batch 16) from source revision
`0331ab743faac6f3e831582e12392ce7982fff21`. The combined training identity is
`a43c848468ad6a2b5f0069aedc34cb41da7d9d4d9f5af77fbb40b7e4cb6f7dcb`; training-time protected-test
access remained zero. Validation comparisons are explicitly non-final and do not select a winner.

## Current sprint

Sprint 5 — campaign `sprint5-final-20260720-v1` is complete and sealed. The 68-image integrity and
leakage preflight passed. Technical `attempt-001` stopped after a result-serialization category-ID
error; its lock and cause are preserved, no partial metric informed a decision, and the unchanged
five-model `attempt-002` completed from synchronized revision `d6fbeea`. All five result hashes and
ten canonical/raw prediction hashes validate. The preregistered ranking recommends `real_only`.

Final protected-test mAP@50-95 is 0.168938 (`synthetic_only`), 0.191505 (`real_25`), 0.198121
(`real_50`), 0.182875 (`real_75`), and 0.211920 (`real_only`). All mixed regimes exceed
synthetic-only, but the non-monotonic `real_75` result prevents a simple causal ratio claim. The
deterministic post-campaign analysis generated 235 tracked metadata selections and 235 ignored
protected-pixel gallery images.

Sprint 6B backend status: `fastapi_service_validated`. `apps/api` exposes health, project, five-model
registry/detail, evaluation/detail, training, reproducibility, reports, single inference, and bounded
batch inference endpoints. It loads one verified checkpoint lazily, selects CPU/CUDA safely, bounds
uploads and parameters, validates MIME and decoding, rejects protected-test SHA-256 values, returns
pixel/normalized boxes and optional annotated PNG data, and never accepts or exposes private server
paths. The API suite passes and a real `real_only` CPU request using a generated non-test 96×64 PNG
returned HTTP 200 with an annotated result.

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

Generated datasets, caches, pretrained/smoke/final weights, returned archives, extracted runs, smoke
outputs, and audits remain ignored.
Versioned manifests, configuration, environment evidence, and documentation are protocol artifacts.
All five full final training experiments now exist and are verified, but no protected-test inference,
final evaluation, final ranking, or operational API exists. The tracked Sprint 4B intake report is
`reports/training/sprint4b_v2_intake_report.md`. The Sprint 6A dashboard foundation still displays
repository metadata and explicit pending/demo states until verified-result integration.

## Next gate

Commit and push the validated FastAPI service without weights. Then export the sealed training,
evaluation, ranking, class, size, latency, campaign, and hash records into the repository dashboard
mode and connect the inference laboratory to the API with no silent demo fallback.
