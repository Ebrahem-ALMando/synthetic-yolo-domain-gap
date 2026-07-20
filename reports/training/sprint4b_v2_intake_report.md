# Sprint 4B V2 Artifact Intake

> **NON-FINAL — VALIDATION SET ONLY.** These values come from the shared 140-image real validation split. They are not protected-test results and do not select the final model.

## Intake verdict

The returned CUDA bundle passed checksum, safe-path, inventory, per-file size/hash, forbidden-content, five-regime identity, configuration equality, CSV integrity, and checkpoint loadability checks. No raw dataset image, secret, final-test output, unsafe symlink, duplicate path, or non-zero test access was found.

- Archive: `sprint4b_training_results.zip` (57286590 bytes)
- Archive SHA-256: `884577a86521ee9fceab997a9ddec2ef9b2a73f8d2673f5bde5f76887a6e905b`
- Inventoried files: 68
- Training identity: `a43c848468ad6a2b5f0069aedc34cb41da7d9d4d9f5af77fbb40b7e4cb6f7dcb`
- Training source revision: `0331ab743faac6f3e831582e12392ce7982fff21`
- Frozen profile: `standard`; batch 16; image size 640
- GPU: `Tesla T4`
- Completed regimes: 5/5, 50 epochs each
- Training-time protected-test access count: 0
- Class order: `fish, jellyfish, penguin, puffin, shark, starfish, stingray`

## Validation-only summary

| Regime | Precision | Recall | mAP@50 | mAP@50-95 | Best epoch | Duration (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_only | 0.60292 | 0.49279 | 0.53750 | 0.30107 | 40 | 603.115 |
| real_25 | 0.68417 | 0.49265 | 0.56542 | 0.32628 | 39 | 589.492 |
| real_50 | 0.65068 | 0.52112 | 0.55067 | 0.31931 | 36 | 574.934 |
| real_75 | 0.65833 | 0.46920 | 0.55570 | 0.33081 | 37 | 557.670 |
| real_only | 0.64256 | 0.47247 | 0.53614 | 0.31250 | 42 | 541.627 |

## Scientific interpretation boundary

The validation comparison is diagnostic evidence that training completed and that all outputs are parseable. It is not a final ranking. The final recommendation remains locked until the Sprint 5 contract is committed and one complete five-model campaign is run on the fixed protected real test split.
