# Experiment Protocol

## Controlled primary regimes

| Regime | Real training images | Synthetic training images |
| --- | ---: | ---: |
| `synthetic_only` | 0% | 100% |
| `real_25` | 25% | 75% |
| `real_50` | 50% | 50% |
| `real_75` | 75% | 25% |
| `real_only` | 100% | 0% |

Whenever dataset availability permits, every primary regime uses an equal total training-image
budget. Sampling rules, replacement policy, and any unavoidable budget deviation must be recorded
before training. The validation protocol, YOLO base model, image size, epochs, batch size,
augmentation policy, optimizer settings, seed, and evaluation thresholds should otherwise remain
controlled unless a change is explicitly part of an experiment.

## Optional ablation

After primary results identify the best mixed regime, `best_mixed_domain_randomized` may compare it
with stronger domain randomization. The real/synthetic proportion and total image budget remain the
same as the selected mixed baseline; only the documented randomization treatment changes.

## Fixed evaluation

All final models are evaluated on the same immutable, held-out real test set. Test images are barred
from training, validation, copy-paste sources, synthetic backgrounds, and synthetic generation.
The test set is not used for model selection or hyperparameter tuning.

## Traceability and reporting

Each run must have a versioned configuration and unique output directory containing machine-written
metadata, logs, checkpoints, and evaluation outputs. Record code revision, dataset/split identity,
seed, environment, command, and timestamps. Reported metrics and tables must be produced from those
outputs by code. No metric may be entered manually, estimated, or invented. Failed or partial runs
must be labeled as such and cannot be presented as completed evidence.

