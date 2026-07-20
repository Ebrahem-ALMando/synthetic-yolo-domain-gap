# Sprint 5 Deterministic Error and Domain-Gap Analysis

This analysis was generated only after the five-model campaign was sealed. It does not alter the preregistered ranking or evaluation thresholds.

## Main finding

`real_only` ranks first at mAP@50-95 0.211920; `synthetic_only` reaches 0.168938. The absolute gap is 0.042982 (25.44% relative to synthetic-only).

Every mixed regime exceeds synthetic-only, so the synthetic data supported non-zero test performance and the mixed sets improved on it. The curve is not monotonic: `real_50` is the strongest mixed regime, while `real_75` falls below both `real_25` and `real_50`. Consequently these results do not establish a causal or universally optimal ratio.

## Sequential marginal changes in mAP@50-95

- `synthetic_only` → `real_25`: +0.022567.
- `real_25` → `real_50`: +0.006616.
- `real_50` → `real_75`: -0.015246.
- `real_75` → `real_only`: +0.029045.

## Per-class and size findings

The leading AP@50-95 regimes by class are: fish `real_50`; jellyfish `real_only`; penguin `synthetic_only`; puffin `real_25`; shark `real_only`; starfish `real_25`; and stingray `real_only`. Penguin is based on only four test images and must not be treated as stable evidence that synthetic-only generalizes better for that class.

`real_only` leads the fixed small- and large-object strata; `real_50` leads medium objects. Small-object AP is weak for every regime, and the custom size-stratified AP is descriptive rather than a COCO area-range metric.

## Error selection

Selections use fixed ordering: highest-confidence false positives/confusions, lowest-IoU matches, hash-ordered small-object misses, largest missed objects, and one highest-IoU success per class and regime. `error_cases.csv` records every selection reason. Gallery bitmaps contain protected test pixels and remain ignored; only metadata is tracked.

## Limitations

The 68-image test split, class imbalance, four-image Penguin coverage, copy-paste synthetic generator, one architecture, and one dataset limit external validity. Differences are associational under this frozen design and are not uncertainty-adjusted causal effects.
