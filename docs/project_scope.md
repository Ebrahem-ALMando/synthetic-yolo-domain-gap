# Project Scope

## Objective

Build a reproducible research pipeline that quantifies how synthetic and real training-data
proportions affect YOLO object-detection performance on real images.

## In scope

- Validate and document a suitable real object-detection dataset.
- Create one immutable real test split and leakage-safe train/validation splits.
- Generate synthetic training data without using test-set content.
- Train controlled YOLO regimes with comparable budgets and settings.
- Evaluate every final model on the same fixed real test set.
- Retain traceability from configuration to output directory and reported result.
- Expose selected trained models through a future FastAPI service and Next.js dashboard.

## Out of scope for Sprint 1

- Dataset download, selection approval, preprocessing, or fake sample creation.
- Synthetic image generation.
- Model download, training, inference, or evaluation.
- Metric tables, plots, or manually supplied results.
- FastAPI endpoint implementation or Next.js initialization.

Aquarium Object Detection is the current candidate, not the selected dataset. Suitability, license,
class definitions, annotation quality, volume, and leakage risks must be validated in Sprint 2.

## Success criteria

The finished project must reproduce each experiment from recorded configuration, prevent test-data
leakage, compare controlled regimes fairly, and support evidence-based conclusions without
fabricated or manually entered metrics.

