# Project State

Last updated: 2026-07-19

## Current sprint

Sprint 3 — Train-Only Object Bank and Reproducible Copy-Paste Synthetic Generator — completed.

## Active real-data contract

- Aquarium Split V2 remains active and immutable: 427 train, 140 validation, and 68 test images.
- Combined real-split identity:
  `02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7`.
- Split V1 remains preserved. V2 frozen hashes and leakage validation pass.
- Raw files remain read-only. Validation/test images are never generator inputs.

## Sprint 3 completed

- Created versioned distribution-matched and unused class-balanced synthetic policies.
- Built a real-train-only bank covering all 3,452 annotations: 3,451 usable records and one explicit
  extremely-small exclusion.
- Recorded 2,361 GrabCut masks and 1,090 feathered rectangular fallbacks without claiming
  segmentation-ground-truth accuracy.
- Visually rejected two candidate revisions and retained them only as ignored diagnostics. The final
  generator uses 2,138 GrabCut sources passing coverage, luminance, connectivity, and fill filters.
- Passed the final deterministic 16-image smoke gate with all seven pasted classes.
- Froze 427 accepted composite images, each retaining every base label and containing at least one
  pasted object. The pool contains 798 pasted and 4,250 total objects.
- Generated complete source/background/image/object-bank/exclusion/failure manifests, a train-only
  YOLO view, quality statistics/charts, and provenance contact sheets.
- Passed train membership, protected Validation/Test path and hash isolation, output collision,
  annotation, pairing, class order, no-overwrite, and exact duplicate checks.
- Reproduced every synthetic image, label, manifest, and combined identity in temporary storage.

## Frozen synthetic identities

- Generator configuration: `7b957f23b46c760e4df446a362a7e1e8f194a54827696880c39c4b905b180eef`
- Object bank: `22d5de79528f5de87b19bae606a93c62af357fc90ad51bfb81e4d197919c54d3`
- Synthetic pool: `3dbd84054e5b2f9d95a3841974cf9c8bd3b987dcd5b84da0be91a06d9b0989ec`

## Scientific limitations

The outputs are copy-paste composites dominated by real-train pixels, not fully synthetic renders.
GrabCut boundaries can be coarse on small, crowded, translucent, dark, or occluded subjects. Some
foregrounds retain background fragments or do not perfectly match base lighting. Quality filtering
changes object eligibility, and actual pasted proportions deviate modestly from finite-sample
targets, especially for rare classes.

## Data and results status

Generated images, crops, masks, smoke diagnostics, and audits remain ignored. Versioned manifests
and configuration are small protocol artifacts. No model weights, training run, inference result,
API, dashboard, or evaluation metric exists.

## Next gate

Sprint 4 may define controlled YOLO experiment configurations using the fixed V2 validation/test
contract and frozen synthetic V1 pool. It must begin with leakage and identity checks. This sprint
did not download or train a model and did not begin Sprint 4.
