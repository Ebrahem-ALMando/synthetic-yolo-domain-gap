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

## 2026-07-18 — Sprint 2 candidate decision

- Conditionally approve Roboflow's Aquarium Combined version 2 (`raw-1024`) based on its official
  638-image description and CC BY 4.0 terms.
- Do not finalize the dataset identifier or numeric class order until the actual export metadata and
  files pass local validation.
- Use the provider's unaugmented 638-image version rather than a generated 4,670-image augmentation
  export, preserving control over future augmentation.
- Require `ROBOFLOW_API_KEY` only through the environment; support checksum-recorded manual ZIP
  import when automatic export is unavailable.
- Preserve raw data, reject silent annotation repairs, and record explicit inclusion/exclusion codes.
- Detect exact duplicates by SHA-256 and near-duplicate candidates by 64-bit dHash with configurable
  Hamming threshold 6; delete nothing automatically.
- Require reviewed source groups before splitting by default. Published collection location/date is
  not sufficient to map individual images to capture sequences.
- Target group-aware 70/20/10 splits with seed 42 and image-level multi-label balancing. Do not call
  the split frozen until real manifests and their hashes exist.
