# Project State

Last updated: 2026-07-18

## Current sprint

Sprint 2 — Candidate Dataset Validation, Acquisition, Audit, and Immutable Real-Data Split.

## Completed

- Created the Python `src` layout and future API/web application boundaries.
- Added central typed YAML configuration with deterministic seed 42.
- Added environment, configuration, directory, test, and static-validation foundations.
- Documented dataset isolation, controlled experiments, reproducibility, and decision history.
- Established ignore rules for data, artifacts, model weights, runs, caches, and secrets.
- Researched the Aquarium candidate from authoritative Roboflow and CC BY 4.0 sources.
- Conditionally approved Aquarium Combined version 2 pending local file audit and provenance review.
- Implemented non-destructive acquisition/import, strict YOLO validation, factual audit statistics,
  SHA-256/dHash duplicate analysis, group-aware splitting, manifest hashing, and leakage checks.

## Blocked or not started

- Dataset download and local validation are blocked because no `ROBOFLOW_API_KEY` or manual export
  is available.
- Actual counts, annotation issues, exclusions, statistics, duplicate groups, and source groups are
  therefore unknown.
- No immutable real split or Aquarium manifests have been generated or frozen.
- Synthetic data generation.
- YOLO model acquisition, training, evaluation, or metric reporting.
- FastAPI inference endpoints and Next.js dashboard.

## Data and results status

No dataset files are present. Aquarium Combined version 2 is conditionally approved, not finally
adopted. No audit outputs, plots, manifests, models, training runs, or metrics exist.

## Next gate

The user must provide `ROBOFLOW_API_KEY` in the local environment or manually download the official
version-2 YOLO export. Sprint 2 must then run validation, audit, duplicate review, source grouping,
split creation, and leakage validation before Sprint 3 can begin.
