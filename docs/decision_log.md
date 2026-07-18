# Decision Log

## 2026-07-18 — Sprint 1 foundation

- Target Python 3.11+ and use a `src` package layout.
- Store typed configuration in `configs/project.yaml` and reject unknown fields to catch mistakes.
- Set deterministic seed 42 and the initial YOLO base model reference to `yolo11n.pt` without
  downloading weights.
- Keep dataset name and class names unresolved until Sprint 2 validation.
- Treat Aquarium Object Detection as a candidate only.
- Define five primary real/synthetic regimes with an equal total image budget when availability
  permits, plus one optional stronger-domain-randomization ablation.
- Freeze and reuse one held-out real test set; prohibit all test content from training, validation,
  and synthetic generation.
- Keep API and web directories as documented placeholders until their implementation sprints.
- Exclude datasets, generated outputs, runs, weights, and secrets from Git.

