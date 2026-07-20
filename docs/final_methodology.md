# Final Methodology

## Study design

The study is a controlled five-regime comparison. Each regime contains 427 training images and uses
the same 140-image real validation set. The regimes contain 0/427, 107/320, 214/213, 320/107, and
427/0 real/synthetic images. Complementary pairing ensures that an underlying real training canvas
appears once per regime, either untouched or as its synthetic derivative.

## Dataset and leakage control

Aquarium Combined v2 was validated locally. Three invalid records were excluded without annotation
repair, leaving 635 accepted images and 4,784 objects across seven ordered classes: fish, jellyfish,
penguin, puffin, shark, starfish, and stingray. Source-aware Split V2 contains 427 train, 140
validation, and 68 protected test images. Its identity is
`02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7`. The Penguin correction
keeps three defensible still-image runs indivisible, leaving only four Penguin test images.

Every build and evaluation preflight verifies manifest hashes and train/validation/test separation.
Test images were never generator inputs, backgrounds, copy-paste sources, training samples, or
validation samples.

## Synthetic generation

The 427-image `aquarium-synthetic-v1` pool is deterministic copy-paste augmentation, not generative
rendering. Every synthetic canvas starts from a real-train image, retains its annotations, and adds
one to three transformed objects from a different training source group. The accepted pool contains
798 pasted objects and 4,250 total annotations. GrabCut-derived masks, quality filtering, bounded
placement, root seed 42, and complete provenance produce identity
`3dbd84054e5b2f9d95a3841974cf9c8bd3b987dcd5b84da0be91a06d9b0989ec`.

## Training

All regimes use YOLO11n initialized from the same base-weight hash, image size 640, 50 epochs,
batch 16, AdamW, the same augmentation configuration, deterministic seed 42, and one frozen Tesla
T4 `standard` profile. Only the training composition differs. Returned archives were checked for
safe paths, per-file hashes, configuration equality, seven-class order, CSV integrity, and loadable
best/last checkpoints. Training-time protected-test access is zero.

## Locked evaluation

Contract `configs/evaluation/sprint5_final.yaml` fixes CPU float32, image size 640, batch 4,
workers 0, confidence floor 0.001, class-aware NMS IoU 0.70, maximum 300 detections, and seed 42.
The primary ranking metric is mAP@50–95, followed for exact ties by mAP@50, macro per-class AP,
Recall, latency, and model size. The contract was committed at `3af03c7` before protected access.

Campaign `sprint5-final-20260720-v1` contains one successful comparable attempt. Attempt 001 failed
on one-based category-ID serialization; partial metrics were barred from decisions. Attempt 002
reran all five models unchanged and sealed five result and ten prediction hashes.

## Analysis

Aggregate, per-class, fixed object-size, latency, confusion, and validation-to-test comparisons were
computed from sealed outputs. Error cases use deterministic rules: highest-confidence false
positives/confusions, lowest-IoU matches, hash-ordered small-object misses, largest missed objects,
largest cross-regime disagreements, and fixed representative successes. Test-image galleries remain
ignored; tracked reports contain metadata only.
