# Train-Only Copy-Paste Synthetic Generation

## Scientific contract

`aquarium-synthetic-v1` is a deterministic copy-paste composite dataset, not rendered imagery and
not output from a generative model. Every one of its 427 images is a newly written JPEG. Each uses
one active Split V2 real-train image as a base, retains every original valid base annotation, and
adds at least one transformed object extracted from a different training source group whenever an
alternative exists. The synthetic-only experiment will use these composite files and no untouched
real image file.

Validation and test manifests are read only by the hard-fail leakage checker. Their images are never
opened for crops, canvases, visual templates, augmentation references, or generator inputs. The
generator refuses a changed Split V2 identity or frozen hash.

## Object bank and masks

The train manifest contains 3,452 objects. The bank records all 3,452: 3,451 usable estimates and
one explicit exclusion for an extremely small crop. Bounding-box-initialized OpenCV GrabCut produced
2,361 masks; 1,090 degenerate/failed attempts produced softly feathered rectangular fallbacks. These
masks are extraction estimates, never segmentation ground truth.

The primary generator uses a quality-filtered subset of 2,138 GrabCut objects. It rejects fallback
crops and masks outside configured coverage, luminance, connected-component, or foreground-fill
bounds. Fallback artifacts remain fully recorded in the bank and contact sheets. Generated crops,
masks, and class-specific context/crop/mask sheets live under ignored processed-data paths.

## Sampling and transformations

The primary pasted-class probabilities come only from real-train object counts: 2,169 fish, 435
jellyfish, 337 penguin, 105 puffin, 254 shark, 54 starfish, and 98 stingray. Sampling is not class
balanced. The unused optional balanced policy is versioned separately.

Each image receives one to three pasted objects. Configured transformations use scale 0.65–1.25,
rotation ±12 degrees, horizontal flip probability 0.5, brightness/contrast/saturation 0.85–1.15,
optional mild blur and noise, JPEG quality 88–96, and alpha feathering 0.5–2 pixels. Vertical flips
are disabled.

Foreground-mask bounds determine pasted annotations. Objects must stay fully inside the canvas,
remain at least four pixels wide/high, occupy at most 45% of either image dimension, have maximum
IoU 0.15, and occlude no existing object by more than 20%. Placement has bounded retries. A failed
sample attempt is recorded and deterministically retried with the next attempt seed.

## Smoke and visual gates

The first smoke pool was rejected because feathered rectangular fallbacks created opaque background
patches. A second smoke/full candidate was rejected because low-light and disconnected masks created
weak dark silhouettes. Both are retained only in ignored diagnostic archives. The final 16-image
smoke gate uses GrabCut-only luminance/connectivity/shape filtering, contains all seven pasted
classes, passes annotation/provenance/leakage validation, and was visually inspected image by image.

The accepted 427-image pool contains 798 pasted objects and 4,250 total annotations. It has no exact
duplicate synthetic image, no exact real-image collision, and all seven classes occur. Representative
base/crop/mask/final sheets cover every class and configured transformation types.

## Provenance and determinism

Manifests under `manifests/aquarium/synthetic/v1` record all repository-relative source, background,
output, hash, seed, transformation, placement, and identity fields. Root seed 42 is expanded into
deterministic sample and placement seeds. Configuration identity, active Split V2 identity, and the
object-bank identity participate in generation seeds.

The pool identity is
`3dbd84054e5b2f9d95a3841974cf9c8bd3b987dcd5b84da0be91a06d9b0989ec`. Non-destructive
temporary reproduction matches the manifests and every image/label hash in Python 3.11.9, Pillow
11.3.0, OpenCV 4.13.0, and NumPy 2.4.6. Observational timestamps are excluded from identity. Codec
or library changes can change JPEG bytes and must be treated as an environment change.

## Limitations

GrabCut boundaries can remain coarse, especially for tiny, crowded, translucent, low-contrast, or
partially occluded subjects. Some composites retain small background fragments or imperfect lighting
agreement between pasted foreground and base canvas. Real-train pixels dominate every base canvas,
so these images measure copy-paste augmentation effects rather than a fully synthetic rendering
domain. The quality filters reduce artifacts but also change which bank objects are eligible.

