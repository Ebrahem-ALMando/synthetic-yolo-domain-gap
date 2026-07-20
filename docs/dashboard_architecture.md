# Dashboard architecture

`apps/web` is an independent Next.js App Router application. It shares no runtime dependency with
the Python training package and does not import datasets, checkpoints, or run outputs. Its only
bridge is the deterministic metadata export created by `scripts/export_dashboard_snapshot.py`.

## Layers

1. Repository export reads authoritative YAML/JSON/CSV metadata and emits safe JSON.
2. `src/types/domain.ts` defines project, identity, dataset, regime, training, evaluation,
   inference, and audit contracts.
3. Repository/demo/API adapters select data explicitly; API mode fails closed.
4. Reusable components provide shell, status, technical text, charts, tables, and workspaces.
5. Explicit App Router pages render each Sprint 6A route.

The snapshot has counts and identities but no image/label paths, pixels, weights, private paths, or
secrets. Test count/protection are visible; protected content is absent. Final metrics remain locked
until validated result archives are integrated.

Sprint 6B may add API mode, result-archive ingestion, model registry, and detection rendering under
the fail-closed contracts in `docs/api_contracts.md`.
