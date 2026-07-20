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

## 2026-07-19 — Sprint 2.5 focused Penguin split-quality review

- Preserve the original 444/128/63 split as immutable Split V1, including its combined identity
  `c926fd840a05385e604682d647b57f2d496c5d31c96f02ad7f4b33eba29b7db4`.
- Review every one of the 71 accepted Penguin images using filename, current-source, MOV,
  viewpoint/background, capture-sequence, and perceptual-similarity contact sheets.
- Treat capture dependency, not shared physical-exhibit appearance, as the grouping criterion.
- Approve three intact Penguin capture runs: IMG_2282-2354, IMG_2519-2530, and IMG_3130-3177.
  Their closest cross-group dHash distance is 15 versus the near-duplicate threshold of 6; no MOV
  identity crosses a boundary, and adjacent captures and confirmed duplicates remain together.
- Reassign the 128 rows in the old broad Penguin exhibit group with their containing run as a direct
  consistency requirement; preserve the other 507 source-review rows unchanged.
- Freeze active Split V2 at 427/140/68 train/validation/test images with Penguin coverage in all
  splits and combined identity `02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7`.
- Accept V2 only after review integrity, all-class coverage, deterministic reproduction, and real
  leakage validation pass. Fix V2 before any Sprint 3 experiment; retain V1 for traceability.

## 2026-07-19 — Sprint 3 train-only copy-paste synthetic V1

- Define the primary 427-image pool as distribution-matched copy-paste compositing, not fully
  rendered or generative imagery. Reuse real-train pixels honestly in provenance.
- Build the bank from all 3,452 V2 train objects: retain 3,451 usable extraction estimates and record
  one extremely small exclusion. Record 2,361 GrabCut and 1,090 feathered fallback outcomes.
- Reject the first smoke gate because rectangular fallbacks produced opaque background patches.
  Reject the first full candidate because some low-light/disconnected masks produced weak dark
  silhouettes. Preserve both only as ignored diagnostics; do not publish their identities.
- Restrict primary sampling to 2,138 quality-filtered GrabCut objects while preserving all bank rows.
  Keep the primary class probabilities equal to the seven real-train object proportions.
- Freeze 427 accepted composites with 798 pasted objects and 4,250 total annotations. Retain all
  3,452 base labels and use all 427 train canvases exactly once in deterministic shuffled order.
- Record object-bank identity
  `22d5de79528f5de87b19bae606a93c62af357fc90ad51bfb81e4d197919c54d3` and pool identity
  `3dbd84054e5b2f9d95a3841974cf9c8bd3b987dcd5b84da0be91a06d9b0989ec`.
- Accept Sprint 3 only after smoke/full visual inspection, annotation and provenance validation,
  exact-collision checks, leakage validation, and non-destructive seed-42 reproduction pass.
- Do not download a model or begin training, inference, API, dashboard, or Sprint 4 work.

## 2026-07-19 — Sprint 4A controlled construction and smoke training

- Freeze five 427-image regimes at 0/427, 107/320, 214/213, 320/107, and 427/0 real/synthetic.
- Use seed-42 multi-label deficit selection; use the complementary synthetic derivative for every
  unselected real canvas so all 427 canvases appear exactly once per regime.
- Freeze experiment-design identity
  `abe47eebc6567de98401e49e75279935cdeb0738558a40ee58dd2b423214ee4c` after manifest,
  protected-set, pairing, hash, and materialized-view validation.
- Fix the shared future configuration at YOLO11n, 640 pixels, 50 epochs, batch 16, AdamW, and the
  documented moderate augmentation policy. Predeclare batch 4 as the only low-memory alternative;
  Sprint 4B must select one profile for every regime before training.
- Classify this machine as `smoke_training_only_cpu`: its MX550 has 2 GiB, while installed PyTorch
  2.13.0 is CPU-only and does not detect CUDA.
- Accept the five-regime technical gate after one epoch on deterministic 16/14 subsets at 320 pixels
  and batch 2. Treat all smoke metrics and checkpoints as non-scientific ignored artifacts.
- Keep the real test set identity-check-only. Do not run full training, test inference, final
  evaluation, failure-case analysis, API, dashboard, or Sprint 5 work.

## 2026-07-19 — Sprint 4B external CUDA execution contract

- Keep local status at `awaiting_external_cuda_execution`; CPU-only PyTorch is not a final-training
  environment and no local 50-epoch run is authorized.
- Transfer a checksummed, inventory-bound bundle containing only required code, frozen contracts,
  training pairs, and real-validation pairs. Permit the test manifest only as protected identity
  evidence and reject any test image by path or content hash.
- Require a successful CUDA tensor operation plus a bounded `real_50` memory run. Freeze batch 16
  only with a conservative free-VRAM margin; otherwise freeze batch 4. Never auto-batch or select a
  different profile per regime.
- Persist and revalidate each regime before beginning the next. Treat failed/interrupted attempts as
  immutable evidence and allocate a fresh run directory for retry.
- Export five validated runs and a machine-generated combined training identity. Label validation
  summaries non-final and defer all real-test evaluation and scientific conclusions to Sprint 5.

## 2026-07-19 — Generated clean-HEAD revision binding

- Reject a committed literal expected-revision value because a commit cannot contain its own stable
  SHA without changing that SHA.
- Make the generated internal bundle inventory the sole revision authority. Bind the clean `main`
  HEAD, branch, and false dirty marker into the bundle identity.
- Keep an optional CLI override only as a matching assertion; never allow it to replace inventory.
- Reject missing/malformed revisions, non-main sources, dirty-source bundles, and mismatched
  overrides before materialization or training.
- Declare the prior dirty-worktree bundle invalid. Set status to
  `awaiting_revision_binding_fix_commit`; build no replacement until this change is committed.

## 2026-07-20 — Bundle v2 runtime descriptor completeness

- Diagnose the post-preflight v1 failure as an omitted frozen synthetic train-only descriptor, not
  a request to validate on synthetic images or to change any regime dataset.
- Include and hash `datasets/processed/aquarium/synthetic/v1/data.yaml` as synthetic identity input.
- Keep all primary YOLO runs bound to regime-specific 427-image train views and the same 140-image
  active real-validation view.
- Generate regime YAMLs without platform-specific absolute paths.
- Require five actual final-mode dry-runs from every freshly extracted bundle before handoff.
- Quarantine v1 as `INVALID-MISSING-SYNTHETIC-YAML` and use v2 filenames and persistent state.

## 2026-07-20 — Sprint 4B returned-artifact acceptance

- Accept the returned v2 archive only after its SHA-256 sidecar, internal/external inventories,
  68 per-file hashes/sizes, safe paths, duplicate/symlink rules, and forbidden-content declarations
  reproduce locally.
- Accept exactly five 50-epoch final regimes from source revision
  `0331ab743faac6f3e831582e12392ce7982fff21`, all bound to profile identity
  `34c3c33d70fbec10863c2616f38c470cc8d66e7ab4a74d1ee77a3e61049f54e1` (Tesla T4,
  standard batch 16, 640 pixels).
- Require every best/last checkpoint to load without inference as an Ultralytics detection model at
  nano scale with 2,591,205 parameters and exact class order `fish`, `jellyfish`, `penguin`,
  `puffin`, `shark`, `starfish`, `stingray`.
- Freeze combined Sprint 4B training identity
  `a43c848468ad6a2b5f0069aedc34cb41da7d9d4d9f5af77fbb40b7e4cb6f7dcb` and retain the returned
  archive, extracted checkpoints, and raw run products as ignored local artifacts.
- Publish only machine-derived validation summaries labelled `NON-FINAL — VALIDATION SET ONLY`.
  Do not name a validation leader as the final winner; Sprint 5 contract freeze remains next.
