# SynthDet: Synthetic Data Augmentation for YOLO Object Detection

## Measuring the Synthetic-to-Real Domain Gap

Final university project report — Computer Vision — 20 July 2026

## Abstract

This project measures how controlled replacement of real training images with deterministic
copy-paste composites affects YOLO11n detection on a fixed real test set. Five equal-budget regimes
were trained with 0%, 25%, 50%, 75%, and 100% real images. Source-aware Split V2, training
configuration equality, zero training-time test access, a preregistered evaluation contract, and a
single sealed five-model campaign protect the comparison. On 68 protected real images, mAP@50–95
ranged from 0.168938 for synthetic-only to 0.211920 for real-only. Real-only is recommended by the
predeclared rule; real-50 is the strongest mixed regime. All mixed regimes outperform
synthetic-only, but their non-monotonic ordering does not support a universal optimal ratio. The
project also delivers a traceable FastAPI inference service, Arabic RTL dashboard, final reports,
and reproducible local demonstration.

## الملخص العربي

يقيس المشروع أثر الاستبدال المضبوط لصور التدريب الحقيقية بصور مركبة حتمية على أداء YOLO11n في
اختبار حقيقي ثابت. دُرّبت خمسة أنظمة متساوية الميزانية بنسب حقيقية من 0% إلى 100%. حقق النظام
الحقيقي فقط mAP@50–95 يساوي 0.211920 مقابل 0.168938 للاصطناعي فقط، وكان النظام المختلط 50%
الأفضل بين الخلطات. جميع الخلطات تجاوزت الاصطناعي فقط، لكن الترتيب غير الرتيب يمنع ادعاء نسبة مثلى
عامة. تحمي الدراسة تقسيمات مجمدة وعقد تقييم مسبق وحملة مختومة، وتوفر لوحة عربية وخدمة استدلال
ومواد إعادة إنتاج.

## 1. Introduction

Object detectors benefit from diverse labelled images, yet real annotation is expensive and may be
scarce for uncommon classes or viewing conditions. Synthetic augmentation can increase apparent
variety, but its utility depends on the gap between generated training appearances and real test
images. This project turns that question into a controlled, auditable experiment.

## 2. Problem statement

The practical question is not whether a detector can fit synthetic composites, but how well models
trained with different real/synthetic proportions transfer to previously unseen real aquarium
imagery when every other controllable factor is fixed.

## 3. Research objective

Measure the synthetic-to-real domain gap for one YOLO11n protocol and determine the recommended
regime under a ranking rule declared before protected-test access.

## 4. Research questions

1. How much real-test performance remains under synthetic-only training?
2. Do mixed regimes improve over synthetic-only under a fixed image budget?
3. Which regime ranks first on protected-test mAP@50–95?
4. How do classes, object sizes, errors, validation generalization, and latency differ?

## 5. Verified background and references

The study uses Roboflow's Aquarium Combined v2 `raw-1024` export, whose project page states CC BY
4.0 and documents the collection context. Architecture and command behaviour are recorded from the
pinned local Ultralytics environment rather than from unverified prose. No new external literature
claim is introduced in this report. The verified project references are the
[dataset datasheet](datasets/aquarium_datasheet.md), [synthetic method](synthetic_generation.md),
[training protocol](training_protocol.md), [evaluation protocol](evaluation_protocol.md), and the
[Roboflow dataset page](https://public.roboflow.com/object-detection/aquarium).

## 6. Dataset

The acquired export contains 638 image/label pairs. Strict validation excluded three invalid
records without repairing annotations, leaving 635 accepted images and 4,784 objects. The class
order is fish, jellyfish, penguin, puffin, shark, starfish, and stingray. Fish dominates the object
distribution; starfish is the smallest class by object count.

## 7. Real-data split protocol

Source-aware Split V2 contains 427 train, 140 validation, and 68 protected test images, allocated as
indivisible visual/source groups with seed 42. The split identity is
`02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7`. Every scientific run
checks that no image or source group overlaps across splits.

## 8. Penguin grouping correction

The focused review separated Penguin sequences into three defensible still-image runs. Their
boundaries use filename discontinuities, intact local sequences, visual review, and perceptual-hash
separation. Split V2 therefore has 52/15/4 Penguin images in train/validation/test. The four-image
test coverage makes Penguin estimates particularly uncertain.

## 9. Synthetic-data generator

`aquarium-synthetic-v1` is deterministic copy-paste augmentation. Each canvas begins with a real
train image, retains every valid base annotation, and receives one to three transformed objects from
a different training source group. Scaling, rotation, horizontal flips, colour adjustments, mild
blur/noise, alpha feathering, bounded overlap, and deterministic retries are recorded per sample.

## 10. Object bank

The bank records all 3,452 real-train objects. Quality-filtered GrabCut masks provide 2,138 eligible
sources; failed or degenerate masks remain audited rather than silently disappearing. Test and
validation images are never opened as object sources or visual references.

## 11. Five experiment regimes

| Regime | Real | Synthetic | Total |
|---|---:|---:|---:|
| `synthetic_only` | 0 | 427 | 427 |
| `real_25` | 107 | 320 | 427 |
| `real_50` | 214 | 213 | 427 |
| `real_75` | 320 | 107 | 427 |
| `real_only` | 427 | 0 | 427 |

Complementary pairing ensures that each underlying training canvas appears once in every regime,
either as real or its synthetic derivative.

## 12. YOLO11n training protocol

Every regime uses the same YOLO11n architecture and base-weight hash, 640-pixel inputs, 50 epochs,
batch 16, AdamW, patience and augmentation settings, deterministic seed 42, and common real
validation set. The only permitted scientific difference is data composition.

## 13. CUDA training environment

All models completed sequentially on one frozen `standard` Tesla T4 profile. Returned metadata,
50-epoch CSVs, best/last checkpoints, environment records, and hashes passed local intake. Best
validation epochs were 40, 39, 36, 37, and 42 in regime order. Training-time test access remained
zero.

## 14. Evaluation protocol

The Sprint 5 contract fixes the 68-image manifest, five checkpoint hashes, class order, CPU float32,
640 pixels, batch 4, workers 0, confidence floor 0.001, class-aware NMS IoU 0.70, maximum 300
detections, and seed 42. Primary ranking is mAP@50–95; exact ties use mAP@50, macro class AP,
Recall, latency, then model size.

## 15. Validation results

Validation mAP@50–95 was 0.30107 (`synthetic_only`), 0.32628 (`real_25`), 0.31931 (`real_50`),
0.33081 (`real_75`), and 0.31250 (`real_only`). These values are non-final. Their leader differs
from the protected-test winner, confirming that winner selection could not stop at validation.

## 16. Final test results

| Rank | Regime | Precision | Recall | mAP@50 | mAP@50–95 |
|---:|---|---:|---:|---:|---:|
| 1 | `real_only` | 0.581228 | 0.389207 | 0.405088 | 0.211920 |
| 2 | `real_50` | 0.534024 | 0.359544 | 0.374255 | 0.198121 |
| 3 | `real_25` | 0.587329 | 0.382942 | 0.375535 | 0.191505 |
| 4 | `real_75` | 0.472026 | 0.334425 | 0.360511 | 0.182875 |
| 5 | `synthetic_only` | 0.528115 | 0.335490 | 0.328609 | 0.168938 |

Figure 1 is `reports/final/figures/figure_01_final_metrics.png`; Figure 2 is the domain-gap curve.

## 17. Per-class results

Class leaders vary: fish `real_50`; jellyfish `real_only`; penguin `synthetic_only`; puffin and
starfish `real_25`; shark and stingray `real_only`. The heatmap in Figure 3 shows that no regime
uniformly dominates every class. Penguin's apparent exception must be read with its four-image
limitation.

## 18. Object-size results

Real-only leads small (0.101209) and large (0.278149) objects; real-50 leads medium (0.166008).
Figure 4 shows weak small-object AP across all regimes. The size calculation is descriptive under
frozen original-pixel thresholds.

## 19. Domain-gap analysis

Real-only exceeds synthetic-only by 0.042982 absolute mAP@50–95 (25.44% relative). All mixed
regimes exceed synthetic-only, indicating useful transfer from the copy-paste pool and improved
performance when real images are introduced. The drop from real-50 to real-75 prevents a monotonic
or causal ratio claim.

## 20. Error analysis

The post-seal analysis records deterministic selections of false positives, false negatives,
class confusions, lowest-IoU matches, small-object misses, large misses, disagreements, and fixed
successes. Frequent aggregate confusions include fish with jellyfish/penguin/shark and shark with
fish. Metadata for 235 selected cases is tracked; gallery images remain ignored because they contain
protected pixels.

## 21. Inference system

FastAPI exposes health, project, model registry, evaluation, training, reproducibility, reports,
single inference, and bounded batch inference. Models load lazily one at a time. The service validates
MIME, size, decoding, parameters, filename, checkpoint hash, timeout, and protected-test content
hashes, and returns pixel/normalized boxes, Arabic/English labels, timing, device, and optional PNG.

## 22. Dashboard architecture

The Next.js App Router application is Arabic-first, RTL-native, responsive, dark/light, and uses
Tajawal. Repository mode uses the generated sealed snapshot; API mode fetches and identity-checks
FastAPI; demo mode is visibly marked. API failure never silently falls back to demo data.

## 23. Reproducibility

Every result traces to frozen manifests, configs, source revisions, training identity, contract,
campaign, checkpoints, and output hashes. Windows and Linux commands are documented. Large data,
weights, predictions, and protected galleries remain local and ignored.

## 24. Limitations

The 68-image test, four Penguin images, one dataset, one architecture, one seed, copy-paste
generator, class imbalance, repeated aquarium scenes, and lack of uncertainty intervals constrain
external validity. See [Final Limitations](final_limitations.md).

## 25. Ethical and scientific safeguards

The dataset is attributed under its stated CC BY 4.0 terms. The project does not imply endorsement
by collection venues. Protected data was isolated; no test-driven training, tuning, or selective
reporting occurred. Weak results and the failed technical attempt remain documented.

## 26. Conclusions

Under this frozen design, real-only is the recommended model. Synthetic-only transfers meaningfully
but leaves a measurable gap. Mixed training reduces that gap, with real-50 strongest among mixed
regimes, yet the relation is not monotonic.

## 27. Future work

Repeat across seeds, architectures, datasets, and stronger synthetic generators; quantify confidence
intervals; improve small-object synthesis; evaluate domain randomization as a separately frozen
ablation; and validate outside the two documented aquarium contexts.

## 28. Final figures

1. Final metrics: `reports/final/figures/figure_01_final_metrics.png`.
2. Domain-gap curve: `figure_02_domain_gap.png`.
3. Per-class heatmap: `figure_03_per_class_heatmap.png`.
4. Object-size comparison: `figure_04_object_sizes.png`.
5. Workflow: `figure_05_workflow.png`.

## 29. Final tables

Machine-readable aggregate, per-class winner, and size-winner tables are under
`reports/final/tables/`. Authoritative full results remain under `reports/evaluation/`.

## 30. Hash and command appendix

See [Final Reproducibility Appendix](final_reproducibility_appendix.md) and
`reports/evaluation/sprint5_hash_report.json`.

## 31. Availability statement

Source, configs, safe reports, dashboard, API, and presentation materials are versioned in Git.
Datasets, checkpoints, raw predictions, protected images, and full local releases are intentionally
not distributed through the repository.
