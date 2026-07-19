# Controlled YOLO Training Protocol

## Frozen inputs

Training is permitted only when the active real Split V2 identity is
`02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7`, the synthetic pool
identity is `3dbd84054e5b2f9d95a3841974cf9c8bd3b987dcd5b84da0be91a06d9b0989ec`, and the experiment
design identity is `abe47eebc6567de98401e49e75279935cdeb0738558a40ee58dd2b423214ee4c`.

Every regime has 427 training images. Mixed regimes use complementary pairing: each underlying
real-train canvas appears exactly once, as either the untouched real sample or its corresponding
synthetic composite. All regimes validate on the same 140 active real-validation images. The 68
real-test images are prohibited in Sprint 4A and are represented only by protected manifest
identities during leakage checks.

## Common final-training configuration

The versioned source is `configs/training/common.yaml`. All regimes use YOLO11n initialized from
`yolo11n.pt`, 640-pixel images, 50 epochs, batch 16, seed 42, deterministic mode, AdamW, initial
learning rate 0.001, weight decay 0.0005, cosine scheduling, three warmup epochs, patience 15, two
workers, and no cache. The initial run never resumes.

The shared moderate augmentation policy is horizontal flip 0.5, HSV hue/saturation/value
0.015/0.4/0.3, scale 0.25, translation 0.10, rotation 5 degrees, mosaic 0.5, and close-mosaic 10.
Vertical flip, Ultralytics copy-paste, and mixup are zero. Settings must not change by regime.

The standard profile is batch 16. A single predeclared low-memory profile uses batch 4. Sprint 4B
must select one profile before any primary run and apply it to all five regimes; per-regime automatic
batch or optimizer selection is forbidden.

## Smoke versus final runs

Smoke mode is technical validation only: one epoch, image size 320, batch 2, zero workers, 16
deterministically selected training images, and 14 deterministically selected real-validation
images. These subsets cover all seven classes. Smoke checkpoints and printed validation values are
not scientific results.

The runner verifies identities, manifest hashes, class order, materialized paths, and test non-use;
records the command, revision, environment, timestamps, resolved configuration, and status; and
allocates a unique ignored directory. Exceptions are recorded as failed and interrupts as
interrupted. Existing runs are never overwritten.

```powershell
python scripts/train_yolo.py real_50 --mode smoke --device cpu --dry-run
python scripts/train_yolo.py real_50 --mode smoke --device cpu
```

Final mode has an additional acknowledgement and was not executed in Sprint 4A:

```powershell
python scripts/train_yolo.py real_50 --mode final --device 0 --confirm-final --profile final_profile.json
python scripts/run_all_regimes.py --mode final --device 0 --confirm-final --profile final_profile.json
```

First final runs begin from the frozen pretrained initialization. A failed or interrupted run is
preserved and restarted in a new directory; implicit resume is not allowed by this protocol.

## Local GPU execution

```powershell
python -m pip install -e ".[dev]"
python scripts/inspect_training_environment.py
python scripts/build_experiments.py --materialize-only
python scripts/validate_experiments.py
python scripts/run_all_regimes.py --mode final --device 0 --confirm-final --dry-run
```

Reserve at least 10 GiB for five final run directories and caches; this is a conservative operational
reservation, not a measured final-run size. Outputs are written below `artifacts/experiments/final`.

## Colab GPU execution and transfer bundle

No upload is automatic. Build a checksummed package only when transfer is required:

```powershell
python scripts/build_training_bundle.py --dry-run
python scripts/build_training_bundle.py
```

The archive contains source/configuration, frozen manifests, real-train and synthetic training
pairs, and real-validation pairs. It contains no real-test images or secrets. Its inventory records
every SHA-256 checksum and the command `python scripts/validate_experiments.py`.

Sprint 4B uses `notebooks/sprint4b_full_training_colab.ipynb`; users edit only the grouped bundle
path, persistent-output path, regime selection, and device variables. The notebook validates its
bundle checksum before
safe extraction, pins the runtime, proves CUDA with a real tensor operation, materializes and
validates the five views, and runs a bounded one-epoch memory preflight on `real_50`. Preflight
metrics are technical evidence and are not scientific results.

Batch 16 is selected only after it completes with at least 512 MiB or 10% total VRAM free,
whichever is larger. Otherwise batch 4 is tried. Failure at batch 4 stops the workflow. The selected
640-pixel profile is frozen once and supplied to every primary run.

After extracting the bundle in Colab, the notebook invokes the restart-safe equivalent of:

```bash
python scripts/colab_train.py \
  --repository /content/synthetic-yolo-domain-gap \
  --regime all \
  --device 0 \
  --persistent-output /content/drive/MyDrive/synthdet/sprint4b \
  --profile /content/drive/MyDrive/synthdet/sprint4b/final_profile.json
```

Each completed run is fully validated, copied through a hidden staging directory to Drive, then
revalidated before its completion-state entry becomes `completed`. Interrupted/failed attempts stay
labeled and a retry receives a new directory. A completed run is skipped only if its recorded
checkpoint/results hashes still match. Finalization produces a five-run completion manifest, a
clearly non-final validation table and figure, and a dataset/secret/test-output-free results archive.

The local status is `awaiting_revision_binding_fix_commit`. After that commit and a validated clean
bundle rebuild, it may advance to external CUDA execution. The real test set is prohibited
throughout Sprint 4B.

### Revision binding

A commit cannot safely contain its own literal SHA: changing the file to that SHA creates a new
commit with a different SHA. Therefore no committed notebook/configuration stores the expected
revision. The bundle builder accepts only a clean `main` worktree, resolves HEAD at build time, and
writes `expected_repository_revision`, `source_branch`, and `source_worktree_dirty: false` into the
generated internal inventory. These generated fields participate in the bundle identity.

After extraction, the notebook validates the inventory, resolves the revision automatically,
displays it, and passes it to the training entry point as a matching assertion. Users do not edit
`EXPECTED_REPOSITORY_REVISION`. The previously produced dirty-worktree bundle is invalid and must
not be uploaded. A new clean bundle must be built only after this revision-binding fix is committed.

### Bundle v2 runtime completeness

The first clean v1 transfer bundle was rejected after Colab preflight because it omitted the frozen
synthetic pool's train-only `data.yaml`. That descriptor is identity/provenance input only; final
training always uses each regime's own portable `data.yaml` with the common 140-image active real
validation set. Bundle v2 includes the frozen synthetic descriptor and validates its recorded hash.
Fresh extraction validation now rematerializes all five views and invokes all five actual final
training entry points in `--dry-run` mode before an archive may be accepted.

Use `aquarium-sprint4b-v2.zip` and a new persistent directory such as
`/content/drive/MyDrive/synthdet/sprint4b-v2`; never reuse v1 completion state.
