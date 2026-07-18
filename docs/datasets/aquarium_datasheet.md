# Aquarium Combined v2 Datasheet

Status: dataset approved from local validation; source/duplicate review and immutable splitting are
still pending.

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

SHA-256 found no exact accepted-image duplicates. A 64-bit dHash threshold of 6 produced 8 pending
two-image groups, with pair distances from 2 to 6. Source analysis proposes 83 groups: 17 explicit
video groups covering 81 frames are confirmed from shared `_MOV` filename bases, while 66 groups
covering 554 still images remain pending.

After review, complete groups will be allocated with seed 42 and image-level multi-label balancing,
targeting 70% train, 20% validation, and 10% test. Manifest hashes will freeze the split identity.

## Known limitations

- Only two aquariums and two collection dates are documented, limiting domain diversity.
- No per-image aquarium identity, scene mapping, or still-image capture provenance is published.
- Images show reflections, occlusion, artificial lighting, repeated backgrounds, and sequences.
- Visual sheets show that some consecutive-number runs span different exhibits; filenames alone are
  insufficient for final grouping.
- The class distribution is strongly imbalanced and the modest total size constrains grouped splits.

## Ethical and responsible use

Use the data consistently with CC BY 4.0 attribution. Do not imply endorsement by Roboflow or either
aquarium. Consider venue, visitor, privacy, publicity, and animal-welfare context when inspecting
images. Models trained on these limited venues must not be represented as reliable species or safety
systems without broader validation.

## Audit outputs and split status

Machine records, issue JSON, statistics, eight data-derived plots, duplicate candidates, source
proposals, and contact sheets exist under `reports/dataset_audit/aquarium`. They are generated and
ignored by Git. Human review remains open, so no train/validation/test manifests or split identity
exist and no real-split leakage result can yet be claimed.

