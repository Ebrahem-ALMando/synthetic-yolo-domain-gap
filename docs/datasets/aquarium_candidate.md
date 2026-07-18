# Aquarium Candidate Dataset Validation

Accessed: 2026-07-18

## Identity and authoritative source

- Exact published name on Roboflow Public Datasets: **Aquarium Dataset**, export **raw-1024**.
- Current Roboflow Universe project name: **Aquarium Combined** (`aquarium-combined`).
- Provider, collector, and annotation publisher: Roboflow.
- Original public source: https://public.roboflow.com/object-detection/aquarium
- Current project page: https://universe.roboflow.com/brad-dwyer/aquarium-combined
- Candidate version: version 2, `raw-1024`, exported on 2020-11-18.
- Published size: 638 images. This is the source-image count, not the 4,670 augmented images shown
  by some generated versions.

Roboflow states that it collected the images at the Henry Doorly Zoo in Omaha on 2020-10-16 and the
National Aquarium in Baltimore on 2020-11-14. Roboflow labeled the images for object detection with
some assistance from SageMaker Ground Truth.

## License and permitted use

Roboflow publishes the images and annotations under Creative Commons Attribution 4.0 International
(CC BY 4.0): https://creativecommons.org/licenses/by/4.0/. The dataset page explicitly permits
personal, commercial, and academic purposes with source acknowledgement. CC BY 4.0 permits sharing
and adaptation, including local processing, provided appropriate attribution is given, the license
is linked, and modifications are indicated. The license provides no warranty and does not eliminate
potential privacy, publicity, or other third-party rights.

Required attribution for this project: identify Roboflow as the dataset provider, link the source
dataset and CC BY 4.0, and state that the project creates normalized working copies and new splits.

## Annotations, classes, and exports

- Task: bounding-box object detection; annotation collection is named `creatures`.
- Published classes: `fish`, `jellyfish`, `penguins`, `sharks`, `puffins`, `stingrays`, `starfish`.
- The acquired export's `data.yaml` establishes this stable numeric order: `0 fish`, `1 jellyfish`,
  `2 penguin`, `3 puffin`, `4 shark`, `5 starfish`, `6 stingray`.
- Available exports published by Roboflow include COCO JSON, COCO-MMDetection, CreateML JSON,
  PaliGemma JSONL, Pascal VOC XML, multiple YOLO TXT variants, TensorFlow CSV/TFRecord, RetinaNet
  CSV, and classification conversions.
- The reproducible acquisition command requests version 2 in `yolov5pytorch` format, which provides
  normalized YOLO detection labels and a YAML class mapping suitable for internal validation.

## Provenance and grouping risk

The two collection locations and dates are published, but no authoritative per-image aquarium,
scene, or capture-group mapping was included. Inspection found 81 explicit `_MOV-n` video frames in
17 filename-confirmed video groups. The remaining 554 still images use numbered `IMG_n` filenames;
67 conservative consecutive-number runs were proposed, but visual sheets show that some long runs
cross different exhibits. Roboflow's existing split placement is not accepted as provenance.

If reliable groups cannot be established from acquired metadata or human review, the limitation must
remain explicit. Split creation requires a reviewed source-group CSV by default; its singleton
fallback requires an explicit flag and cannot be described as reliable provenance.

## Access and authentication

The archive was manually acquired through the official Roboflow export interface. Acquisition
metadata records SHA-256 `0c5319224d1862dd7a0b3f6039d520050d91d9de0d075033d97990ff4d2f5eae`;
the independently recomputed hash matches. All 1,281 stored raw files are read-only.

## Suitability assessment

Local validation discovered the published 638 images and 638 matching labels. It accepted 635 images
and excluded 3 without repair: two have zero-area shark rows and one has an empty label/no valid
objects. No corrupt images, missing pairs, unknown classes, malformed rows, out-of-bounds boxes, or
duplicate stems were found. The accepted subset contains all seven classes and 4,784 valid objects.

The file-level evidence supports adoption for this study. Source-group and dHash candidate review
remain mandatory before an immutable split can be created; approval of the dataset does not imply
that a test manifest has been frozen.

APPROVED
