# Project State

Last updated: 2026-07-18

## Current sprint

Sprint 1 — Project Foundation and Experimental Protocol.

## Completed

- Created the Python `src` layout and future API/web application boundaries.
- Added central typed YAML configuration with deterministic seed 42.
- Added environment, configuration, directory, test, and static-validation foundations.
- Documented dataset isolation, controlled experiments, reproducibility, and decision history.
- Established ignore rules for data, artifacts, model weights, runs, caches, and secrets.

## Not started

- Dataset validation, selection, download, splitting, or preprocessing.
- Synthetic data generation.
- YOLO model acquisition, training, evaluation, or metric reporting.
- FastAPI inference endpoints and Next.js dashboard.

## Data and results status

No dataset is selected or present. Aquarium Object Detection is a candidate pending Sprint 2
validation. No model has been downloaded or trained. No metrics, result tables, or figures exist.

## Next sprint gate

Sprint 2 may validate candidate datasets and design the immutable, leakage-safe real split. It must
not treat the current candidate as selected until source, license, annotations, grouping, quality,
and feasibility have been documented.

