# SynthDet — Final Executive Summary

SynthDet investigates how controlled mixtures of real and copy-paste synthetic training images
affect YOLO11n object detection on unseen real aquarium imagery. The experiment fixes the model,
training-image budget (427), real validation set (140 images), training schedule (50 epochs), seed
(42), and all optimization settings. The only intended scientific difference is the real/synthetic
composition: 0%, 25%, 50%, 75%, or 100% real training images.

All five models completed on one frozen Tesla T4 profile. A preregistered contract was committed and
pushed before the protected 68-image real test was opened. One technical attempt failed during
serialization and was excluded from decisions; the unchanged complete retry evaluated all five
models once and sealed every prediction and result with SHA-256.

The preregistered primary metric, real-test mAP@50–95, ranks the models as `real_only` (0.211920),
`real_50` (0.198121), `real_25` (0.191505), `real_75` (0.182875), and `synthetic_only` (0.168938).
Real-only exceeds synthetic-only by 0.042982 absolute, or 25.44% relative. Every mixed regime
outperforms synthetic-only, but the curve is non-monotonic; therefore the study does not establish a
universal optimal mixture or a causal dose-response relationship.

Real-only leads small and large objects; real-50 leads medium objects. Per-class winners vary, and
the synthetic-only Penguin lead is highly uncertain because Penguin occurs in only four test images.
The main practical recommendation is `real_only` under the frozen ranking rule, while `real_50` is
the strongest mixed regime and illustrates that synthetic replacement can preserve substantial
performance when real training data is limited.

The deliverable includes a traceable scientific pipeline, sealed reports, Arabic RTL dashboard,
FastAPI model registry and protected-hash-aware inference service, reproducibility records, final
presentation, and local demo tooling. Model weights and all protected test pixels remain untracked.
