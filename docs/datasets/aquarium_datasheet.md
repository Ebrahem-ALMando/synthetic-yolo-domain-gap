# Aquarium Dataset Preliminary Datasheet

Status: pre-acquisition candidate datasheet; no local file audit has been completed.

## Motivation

The candidate provides real aquarium imagery for measuring how synthetic/real training mixtures
affect YOLO detection on a fixed real test set.

## Composition

Roboflow publishes 638 source images with bounding-box labels for seven creature categories. Local
image counts, object counts, resolutions, class balance, empty-label counts, and exclusions are
unknown until acquisition and must not be inferred from previews or generated dataset versions.

## Source and license

Roboflow reports collection at the Henry Doorly Zoo in Omaha and the National Aquarium in Baltimore
on two dates in 2020, followed by Roboflow/SageMaker Ground Truth annotation. The source is
https://public.roboflow.com/object-detection/aquarium and the stated license is CC BY 4.0. Academic
use and local adaptation are permitted with attribution, a license link, and disclosure of changes.

## Annotation format and class definitions

The task is object detection. Roboflow publishes the class names fish, jellyfish, penguins, sharks,
puffins, stingrays, and starfish and offers YOLO, COCO, Pascal VOC, and other formats. Stable numeric
class order remains pending validation against the acquired `data.yaml`.

## Known limitations

- Only two aquariums and two collection dates are documented, limiting domain diversity.
- No per-image source, scene, or sequence mapping is currently published.
- The modest dataset size may limit group-stratified class coverage.
- Aquarium viewing conditions may include reflections, occlusion, artificial lighting, and repeated
  backgrounds; their actual frequency has not been audited.
- Annotation quality, corrupt files, duplicates, empty labels, and imbalance remain unmeasured.

## Cleaning decisions and exclusions

None have been made. The validator never silently repairs annotations. Invalid rows and files will
receive explicit issue codes and exclusion reasons. Any future safe repair must be opt-in, preserve
raw data, and record before/after values; no repair capability is currently enabled.

## Split methodology

After validation, SHA-256 exact matches and 64-bit difference-hash candidates (default Hamming
distance at most 6) will be reviewed. Reviewed duplicate and source groups will be allocated as
units using deterministic seed 42 and image-level multi-label class membership, targeting 70% train,
20% validation, and 10% test. Manifest hashes will freeze the split identity.

## Ethical and responsible use

Use the data for research consistent with CC BY 4.0 attribution. Do not imply endorsement by
Roboflow or either aquarium. Consider venue, visitor, privacy, publicity, and animal-welfare context
when inspecting images. Models trained on these limited venues should not be represented as reliable
species-recognition or safety systems without broader validation.

## Audit status

Acquisition is blocked on a user-provided Roboflow API key or manual version-2 YOLO export. No audit
statistics, plots, exclusions, duplicate groups, source groups, or frozen manifests currently exist.

