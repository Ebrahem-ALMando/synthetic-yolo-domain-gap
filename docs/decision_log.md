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

## 2026-07-18 — Sprint 2 real-data execution

- Approve Aquarium Combined version 2 after the archive identity, read-only raw storage, stable class
  order, and local file-level validation matched the published source.
- Accept 635 images and exclude 3 complete records without repair: two for zero-area boxes and one
  for an empty label/no valid objects.
- Use only accepted-image statistics: 4,784 objects across all seven validated classes.
- Keep all 8 dHash groups pending formal human review even though visual inspection supports their
  interpretation as adjacent or near-identical captures; there are no SHA-256 exact duplicates.
- Confirm 17 `_MOV` groups from explicit shared filename bases. Keep 66 still-image source proposals
  pending because consecutive numbers do not prove common scenes and some contact sheets cross
  exhibits.
- Strengthen split creation to reject pending duplicate and source review statuses. Do not create
  manifests, hashes, a test split, or a leakage result until the review gate is resolved.

## 2026-07-18 — Sprint 2 review resolution and immutable split

- Confirm all 8 dHash pairs after inspecting the actual sheets; retain both images and treat each
  pair as indivisible.
- Inspect all 58 source-sheet pages and all 22 pending singleton images. Split seven original groups
  at visible scene changes, merge repeated-background/same-exhibit groups conservatively, retain 29
  groups unchanged, and approve six true isolated images as `not_applicable` singletons.
- Use 52 final stable source groups. Prefer source/background isolation over exact ratios or complete
  class coverage.
- Correct the greedy allocator to score the global three-split objective; the previous local score
  could place every group in train for this grouped dataset.
- Freeze 444 train, 128 validation, and 63 test images using seed 42. All 71 `penguin` images remain
  train-only because their repeated enclosure is one indivisible reviewed source group.
- Record combined split identity
  `c926fd840a05385e604682d647b57f2d496c5d31c96f02ad7f4b33eba29b7db4`; reproduce it in
  temporary storage without overwriting frozen files.
- Require every frozen path to be repository-relative. Preserve the first otherwise-valid
  raw-export-relative candidate under the ignored audit directory and freeze a corrected manifest
  set rather than overwriting files in place.
- Accept the split only after review-integrity and real-leakage checks pass. The test manifest is now
  fixed and prohibited from training, validation, copy-paste, synthetic backgrounds, augmentation,
  or any synthetic-generation reference.
