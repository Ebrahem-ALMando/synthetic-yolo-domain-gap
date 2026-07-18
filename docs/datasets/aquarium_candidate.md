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
- The stable numeric order will be accepted only from the downloaded export's `data.yaml`; this
  document does not substitute a remembered or inferred order.
- Available exports published by Roboflow include COCO JSON, COCO-MMDetection, CreateML JSON,
  PaliGemma JSONL, Pascal VOC XML, multiple YOLO TXT variants, TensorFlow CSV/TFRecord, RetinaNet
  CSV, and classification conversions.
- The reproducible acquisition command requests version 2 in `yolov5pytorch` format, which provides
  normalized YOLO detection labels and a YAML class mapping suitable for internal validation.

## Provenance and grouping risk

The two collection locations and dates are published, but no authoritative per-image aquarium,
video, scene, sequence, or capture-group mapping has been found. Filenames and export structure must
be inspected after acquisition. Roboflow's pre-existing train/validation/test placement is not
accepted as evidence of sequence-safe grouping and will not be reused blindly.

If reliable groups cannot be established from acquired metadata or human review, the limitation must
remain explicit. Split creation requires a reviewed source-group CSV by default; its singleton
fallback requires an explicit flag and cannot be described as reliable provenance.

## Access and authentication

The metadata and previews are public. Roboflow's documented export API requires a private API key;
the Universe download interface also presents sign-in/sign-up controls. No key is present in the
current environment. The repository uses `ROBOFLOW_API_KEY` and never stores or prints its value.
Official export documentation: https://docs.roboflow.com/developer/export-data

## Suitability assessment

The license permits academic use and transformations with attribution, object-detection annotations
and YOLO exports are available, and 638 source images are sufficient for a modest controlled
university experiment. Approval remains conditional because the files, annotation integrity,
actual class order, duplicate rate, class coverage, and per-image provenance have not been audited.
No immutable test split may be declared until those gates pass.

CONDITIONALLY APPROVED
