# Aquarium Combined v2 Datasheet

Status: dataset approved, visually reviewed, and frozen into immutable seed-42 manifests.

## Motivation

The dataset provides real aquarium imagery for measuring how synthetic/real training mixtures affect
YOLO detection on a fixed real test set.

## Source and license

Roboflow reports collection at the Henry Doorly Zoo in Omaha and the National Aquarium in Baltimore
on two dates in 2020, followed by Roboflow/SageMaker Ground Truth annotation. The source is
https://public.roboflow.com/object-detection/aquarium and the stated license is CC BY 4.0. Academic
use and local adaptation are permitted with attribution, a license link, and disclosure of changes.

The acquired version-2 `raw-1024` YOLOv5 PyTorch archive has SHA-256
`0c5319224d1862dd7a0b3f6039d520050d91d9de0d075033d97990ff4d2f5eae`.

## Locally measured composition

The published count is 638 images. The acquired export contains exactly 638 supported images and 638
matching labels. Strict validation accepts 635 images and excludes 3. Accepted images contain 4,784
valid objects. Widths and heights range from 576 to 1024 pixels; aspect ratios range from 0.5625 to
1.7778, with mean 0.8849 and median 0.75.

| Class ID | Class | Objects | Images containing class |
| ---: | --- | ---: | ---: |
| 0 | fish | 2,645 | 331 |
| 1 | jellyfish | 694 | 52 |
| 2 | penguin | 516 | 71 |
| 3 | puffin | 284 | 66 |
| 4 | shark | 345 | 174 |
| 5 | starfish | 116 | 59 |
| 6 | stingray | 184 | 124 |

The maximum-to-minimum nonzero object-count ratio is 22.8017 (`fish` to `starfish`). Accepted images
contain 1 to 56 objects, with mean 7.5339 and median 5.

## Annotation format and class definitions

The task is YOLO object detection. The acquired `data.yaml`, not published prose or memory, defines
the stable order: `fish`, `jellyfish`, `penguin`, `puffin`, `shark`, `starfish`, `stingray`.

## Validation and exclusions

No annotations were repaired. Three images are excluded as complete records:

- `test/...IMG_2423...`: `non_positive_box` at label line 17.
- `test/...IMG_2570...`: `non_positive_box` at label line 15.
- `train/...IMG_3133...`: `empty_label` and `no_valid_objects`.

The first two images contain other valid rows, but the strict protocol excludes the entire image to
avoid silently changing source annotations. Included-object statistics omit all boxes from those
excluded records.

The project size rule counts 508 small, 2,517 medium, and 1,759 large accepted objects. Box area
ratios range from 0.0000012716 to 0.7265917460, with mean 0.0229023 and median 0.00762177.

## Duplicate and source analysis

SHA-256 found no exact accepted-image duplicates. A 64-bit dHash threshold of 6 produced 8 two-image
groups, with pair distances from 2 to 6. Agent-assisted inspection of every duplicate sheet confirmed
all eight as adjacent or near-identical captures; no image was deleted.

The source review inspected all 58 generated source-sheet pages and 22 pending singleton images.
Seven original groups were split at visible scene boundaries, 42 participated in conservative
same-exhibit merges, 29 remained unchanged, and 6 isolated images became `not_applicable`
singletons. All 635 accepted images now occur exactly once in 52 stable source groups.

Split V1 used deterministic image-level multi-label allocation of indivisible groups with seed 42
and produced 444/128/63 train/validation/test images. It remains byte-for-byte preserved.

The focused Sprint 2.5 review inspected all 71 Penguin images and separated same-exhibit appearance
from capture dependency. Three intact still-image runs were defensible: IMG_2282-2354 (52 Penguin
images/337 objects), IMG_2519-2530 (4/20), and IMG_3130-3177 (15/159). There are no explicit MOV IDs;
the minimum cross-group dHash distance is 15 against threshold 6, and confirmed duplicate pairs stay
within a run. For consistency, all 128 images in the old broad source group were assigned with their
containing run; the other 507 source assignments were unchanged.

Active Split V2 contains 427 train images (67.24%), 140 validation images (22.05%), and 68 test
images (10.71%). Source-group counts are 41, 8, and 5; object counts are 3,452, 717, and 615.
Penguin coverage is respectively 52/337, 15/159, and 4/20 images/objects.

## Known limitations

- Only two aquariums and two collection dates are documented, limiting domain diversity.
- No per-image aquarium identity, scene mapping, or still-image capture provenance is published.
- Images show reflections, occlusion, artificial lighting, repeated backgrounds, and sequences.
- Visual sheets show that some consecutive-number runs span different exhibits; filenames alone are
  insufficient for final grouping.
- The class distribution is strongly imbalanced and the modest total size constrains grouped splits.
- Per-image venue/date provenance is still unavailable. The three Penguin boundaries rely on
  filename discontinuities, intact local sequences, visual passes, and perceptual-hash separation.
- The shortest Penguin group contains only four images, so test estimates for Penguin will have high
  sampling uncertainty even though the class is now measurable.

## Ethical and responsible use

Use the data consistently with CC BY 4.0 attribution. Do not imply endorsement by Roboflow or either
aquarium. Consider venue, visitor, privacy, publicity, and animal-welfare context when inspecting
images. Models trained on these limited venues must not be represented as reliable species or safety
systems without broader validation.

## Audit outputs and split status

Machine records, issue JSON, statistics, review logs, contact sheets, and split audit figures exist
under `reports/dataset_audit/aquarium`; generated outputs are ignored by Git. Frozen manifests live
under `manifests/aquarium`. Split V1 lives under `manifests/aquarium/v1` with combined identity
`c926fd840a05385e604682d647b57f2d496c5d31c96f02ad7f4b33eba29b7db4`. Active Split V2 lives
under `manifests/aquarium/v2` with combined identity
`02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7`. Both reproduce exactly
with seed 42; V2 passes the hard-fail real leakage validation. Synthetic-source and
synthetic-background manifests do not exist; their absence is not evidence of generation.
