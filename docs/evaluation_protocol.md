# Sprint 5 Final Evaluation Protocol

## Status and scope

This protocol preregisters the only authorized protected real-test campaign for SynthDet. Its
machine-readable authority is `configs/evaluation/sprint5_final.yaml`. The contract is frozen and
must be committed and pushed before any protected test image, label, prediction, or metric is read.
The contract records the source revision from which it was generated; the later campaign lock must
also record the exact commit that introduced the contract, avoiding a self-referential commit hash.

The protected set is immutable Aquarium Split V2: 68 unique real images, identity
`02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7`, manifest SHA-256
`02e133b93f840ef95044c75e3bab0a6fec19f62ac1d708993aab536747952c52`. Exactly five already trained
`best.pt` checkpoints participate. No checkpoint may be altered or retrained.

## Shared configuration

All five models use Ultralytics 8.4.101 with PyTorch 2.13.0+cpu, image size 640, batch 4, CPU,
workers 0, seed 42, deterministic execution, float32, confidence floor 0.001, IoU/NMS threshold
0.70, class-aware NMS, and maximum 300 detections. Augmented inference, class-agnostic NMS,
single-class mode, automatic batching, half precision, and model-specific overrides are forbidden.
Batch 4 and workers 0 are pre-access operational choices for the documented CPU-only Windows
environment; they apply identically to all regimes and were not chosen from test behavior.

The confidence floor is the already frozen training-validation floor used to integrate the full
precision-recall curve; it is not an interactive deployment threshold. Thresholds, preprocessing,
image size, NMS, class mappings, precision, device, and worker policy cannot change after results are
observed.

## Primary metric and final recommendation

The final recommendation is the model with the highest `mAP@50-95` on the fixed real test set. Exact
ties are broken, in order, by higher `mAP@50`, higher macro per-class AP@50-95, higher Recall, lower
measured latency, then lower checkpoint size. No validation-only result can select the final model,
and no winner is declared until complete comparable outputs exist for all five regimes.

## Object-size policy

Object size uses ground-truth bounding-box area in original-image pixels, matching the dataset audit:

- small: area below 32² (1,024) pixels;
- medium: area from 1,024 inclusive to 96² (9,216) exclusive;
- large: area at least 9,216 pixels.

These thresholds cannot be redefined after the campaign.

## One locked campaign and retry policy

Campaign `sprint5-final-20260720-v1` is the only successful evaluation campaign authorized. Before
starting, the runner must verify the committed contract, its SHA-256, all five checkpoint hashes,
the manifest hash, 68 unique image-label pairs, exact class order, and absence of train/validation
overlap, then create `reports/evaluation/sprint5_campaign_lock.json`.

A technical retry is allowed only when an infrastructure or execution failure prevents a complete,
comparable five-model result. The failed attempt, logs, partial files, cause, and unchanged contract
hash must be preserved. Partial metrics must not be used to adjust anything. A retry receives a new
attempt ID and reruns all five models under the identical contract; model-specific or tuned retries
are forbidden.

## Predictions, latency, and error analysis

Each model receives one Ultralytics validation pass. Raw predictions are cached deterministically
and hashed for later analysis. Latency comes from the same validation pass as per-image average
preprocess, inference, and postprocess time; loading is excluded, framework warm-up uses only a
synthetic tensor, and protected images are not repeated for latency benchmarking. Throughput is
1,000 divided by total measured milliseconds per image.

Predictions may be used for deterministic error analysis only after the complete five-model campaign
is sealed. Selection rules must be declared in code/metadata (for example confidence-ordered false
positives, false negatives, lowest-IoU matches, class confusions, small-object misses, and
cross-regime disagreements). Protected pixels and galleries remain ignored and are never committed.

## Limitations

The 68-image test set is adequate for one controlled course-project comparison but is small for
strong external-validity claims. Per-class estimates can be unstable, particularly Penguin, which
appears in only four test images. Results characterize this frozen Aquarium Split V2, these five
checkpoints, and this preprocessing/runtime contract; they do not establish universal causal effects
for other domains, generators, architectures, hardware, or datasets.
