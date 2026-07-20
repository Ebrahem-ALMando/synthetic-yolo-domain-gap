# Final Results

## Aggregate protected-test metrics

| Rank | Regime | Precision | Recall | mAP@50 | mAP@50–95 |
|---:|---|---:|---:|---:|---:|
| 1 | `real_only` | 0.581228 | 0.389207 | 0.405088 | 0.211920 |
| 2 | `real_50` | 0.534024 | 0.359544 | 0.374255 | 0.198121 |
| 3 | `real_25` | 0.587329 | 0.382942 | 0.375535 | 0.191505 |
| 4 | `real_75` | 0.472026 | 0.334425 | 0.360511 | 0.182875 |
| 5 | `synthetic_only` | 0.528115 | 0.335490 | 0.328609 | 0.168938 |

The preregistered recommendation is `real_only`. Relative to synthetic-only, its mAP@50–95 gain is
0.042982 absolute and 25.44% relative. Sequential changes are +0.022567 (0→25%), +0.006616
(25→50%), −0.015246 (50→75%), and +0.029045 (75→100%). The non-monotonic sequence precludes a
general optimum claim.

## Validation versus test

Validation mAP@50–95 values were 0.30107, 0.32628, 0.31931, 0.33081, and 0.31250 in regime order.
They are explicitly non-final and did not select a winner. Their leader (`real_75`) differs from the
test winner (`real_only`), demonstrating why the final recommendation could not be based on
validation alone.

## Class and size findings

Per-class AP@50–95 leaders are: fish `real_50`; jellyfish `real_only`; penguin
`synthetic_only`; puffin `real_25`; shark `real_only`; starfish `real_25`; and stingray
`real_only`. Penguin evidence is unstable because only four protected test images contain it.

Under the frozen original-pixel area strata, `real_only` leads small objects (0.101209) and large
objects (0.278149), while `real_50` leads medium objects (0.166008). Small-object AP remains weak
for every regime. These custom descriptive strata are not COCO area-range metrics.

## Runtime

Measured CPU total latency per image is 50.99 ms (`synthetic_only`), 66.20 ms (`real_25`),
68.66 ms (`real_50`), 66.10 ms (`real_75`), and 68.71 ms (`real_only`). Loading is excluded;
these CPU measurements do not estimate GPU production latency.

Figures are generated under `reports/final/figures/`; machine-readable tables are under
`reports/final/tables/`. The authoritative detailed sources remain `reports/evaluation/`.
