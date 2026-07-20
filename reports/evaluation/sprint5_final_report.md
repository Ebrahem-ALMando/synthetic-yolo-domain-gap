# Sprint 5 Final Protected-Test Evaluation

## Campaign integrity

Campaign `sprint5-final-20260720-v1` evaluated the fixed 68-image real Split V2 with one shared
contract: image size 640, batch 4, CPU float32, workers 0, seed 42, confidence floor 0.001,
class-aware NMS IoU 0.70, and maximum 300 detections. The contract was frozen and pushed before
protected access. Checkpoint, manifest, prediction, and result hashes validate.

`attempt-001` failed after one model because the output serializer did not normalize Ultralytics'
one-based non-COCO JSON category IDs. The partial result was barred from decisions. After an
infrastructure-only mapping fix was committed, `attempt-002` reran all five models unchanged and
completed successfully.

## Aggregate results and preregistered ranking

| Rank | Regime | Precision | Recall | mAP@50 | mAP@50-95 |
|---:|---|---:|---:|---:|---:|
| 1 | `real_only` | 0.581228 | 0.389207 | 0.405088 | 0.211920 |
| 2 | `real_50` | 0.534024 | 0.359544 | 0.374255 | 0.198121 |
| 3 | `real_25` | 0.587329 | 0.382942 | 0.375535 | 0.191505 |
| 4 | `real_75` | 0.472026 | 0.334425 | 0.360511 | 0.182875 |
| 5 | `synthetic_only` | 0.528115 | 0.335490 | 0.328609 | 0.168938 |

The preregistered primary metric recommends `real_only`. Its mAP@50-95 exceeds synthetic-only by
0.042982 absolute, or 25.44% relative to the synthetic-only value. Every mixed regime exceeds
synthetic-only, while `real_50` is the strongest mixed regime. The sequence is not monotonic:
`real_75` falls below `real_25` and `real_50`, so no universal optimal mixing ratio or causal effect
should be inferred.

## Class and object-size findings

The highest class AP@50-95 belongs to `real_50` for fish, `real_only` for jellyfish, `synthetic_only`
for penguin, `real_25` for puffin and starfish, and `real_only` for shark and stingray. Penguin's
apparent synthetic-only lead is especially uncertain because only four protected test images
contain that class.

Under the frozen original-pixel area thresholds, `real_only` leads small objects (0.101209) and
large objects (0.278149), while `real_50` leads medium objects (0.166008). Small-object performance
is weak across every regime. These custom descriptive AP strata are not interchangeable with COCO's
area-range implementation.

## Runtime and limitations

CPU total latency ranges from 50.99 to 68.71 ms/image in the same validation pass. These numbers
exclude model loading and do not predict GPU latency. The 68-image set, class imbalance, four-image
Penguin coverage, copy-paste synthetic generator, single architecture, and single source dataset
limit precision and external validity. Full deterministic error metadata is reported separately in
`reports/analysis/error_analysis.md`; protected gallery pixels remain ignored.
