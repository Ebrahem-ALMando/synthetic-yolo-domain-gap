# Aquarium synthetic manifests

`v1/` freezes the train-only `aquarium-synthetic-v1` copy-paste pool. Its CSVs record every base
canvas, pasted object, object-bank record, explicit exclusion, generated image/label hash, and failed
sample attempt. `generation_metadata.json` records the active real split, configuration, object-bank
identity, environment, manifest hashes, and combined pool identity.

Generated images, labels, crops, masks, and previews remain ignored under `datasets/processed`.
Reproduction must use `scripts/generate_synthetic.py --verify-frozen`; frozen files must not be
overwritten.

