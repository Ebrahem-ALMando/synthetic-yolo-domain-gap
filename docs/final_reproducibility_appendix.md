# Final Reproducibility Appendix

## Frozen identities

| Artifact | Identity |
|---|---|
| Real Split V2 | `02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7` |
| Synthetic generator config | `7b957f23b46c760e4df446a362a7e1e8f194a54827696880c39c4b905b180eef` |
| Object bank | `22d5de79528f5de87b19bae606a93c62af357fc90ad51bfb81e4d197919c54d3` |
| Synthetic pool | `3dbd84054e5b2f9d95a3841974cf9c8bd3b987dcd5b84da0be91a06d9b0989ec` |
| Experiment design | `abe47eebc6567de98401e49e75279935cdeb0738558a40ee58dd2b423214ee4c` |
| Training | `a43c848468ad6a2b5f0069aedc34cb41da7d9d4d9f5af77fbb40b7e4cb6f7dcb` |
| Evaluation contract | `ef826f5eb104171039e3029b0600a091e8171a1ad6e9bd8d771b2bb5645c407c` |
| Test manifest | `02e133b93f840ef95044c75e3bab0a6fec19f62ac1d708993aab536747952c52` |
| Campaign | `sprint5-final-20260720-v1` / `attempt-002` |

Checkpoint, result, and prediction hashes are recorded in
`reports/evaluation/sprint5_hash_report.json`; training intake hashes are in
`reports/training/sprint4b_v2_hash_report.json`.

## Core commands

```bash
python -m pip install -e ".[dev]"
python scripts/check_environment.py
python -m pytest -q
python -m ruff check .
python scripts/validate_sprint5_outputs.py
python scripts/generate_final_assets.py
```

Frontend:

```bash
cd apps/web
npm ci
npm run snapshot:validate
npm run typecheck
npm run lint
npm run test
npm run build
```

Backend on Windows PowerShell:

```powershell
$env:PYTHONPATH="src;apps/api"
python -m uvicorn synthdet_api.main:app --host 127.0.0.1 --port 8000
```

Backend on Linux/macOS:

```bash
PYTHONPATH=src:apps/api python -m uvicorn synthdet_api.main:app --host 127.0.0.1 --port 8000
```

## Access audit

Training test access is zero. One protected campaign was authorized after contract commit
`3af03c7`. Attempt 001 was a recorded technical failure with zero complete comparable models and no
partial result used for a decision. Attempt 002 completed all five unchanged models. The test was not
used for retraining, threshold tuning, model-specific settings, demo fixtures, or frontend assets.

Large inputs, weights, returned archives, extracted runs, prediction caches, gallery pixels, and full
local release archives are ignored by Git. Reproducers must supply checkpoints whose hashes match the
frozen contract.
