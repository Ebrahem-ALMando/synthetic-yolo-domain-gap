# SynthDet Arabic dashboard

Sprint 6A provides an isolated, Arabic-first Next.js dashboard for the scientific repository. It
does not train, evaluate, or read image pixels. Repository mode consumes only the generated,
metadata-only snapshot in `src/data/generated/project-snapshot.json`.

## Setup and validation

```bash
cd apps/web
npm install
npm run snapshot:export
npm run dev
npm run validate
```

Open `http://localhost:3000`. `npm` is used because the repository had no package manager and
`pnpm` was not installed; `package-lock.json` freezes the resolved graph.

## Environment and data modes

Copy `.env.example` to `.env.local` only for overrides.

- `NEXT_PUBLIC_DATA_MODE=repository`: authoritative generated metadata; default.
- `NEXT_PUBLIC_DATA_MODE=demo`: visibly labelled presentation values from `src/data/demo.ts`.
- `NEXT_PUBLIC_DATA_MODE=api`: reserved for Sprint 6B and hard-fails until an API exists.
- `NEXT_PUBLIC_APP_URL`: canonical URL for metadata.

API mode never falls back. Repository mode never exposes raw paths, protected content, weights, or
final metrics.

## Snapshot regeneration

From the repository root run `python scripts/export_dashboard_snapshot.py`. The exporter reads
authoritative project/training YAML, frozen metadata JSON, and metadata CSV files. It emits counts
and identities only. `npm run snapshot:validate` rejects raw paths, weights, protected content, or
a non-zero test-access count.

## Official logo

The UI references `public/brand/synthdet-logo.png` in the shell, overview, about page, and metadata.
The exact official user-supplied 2024 x 2024 bitmap is stored there without recompression or
redrawing. Its SHA-256 is
`bead045dcc98d5f8fc205542216b720453f62690d959280a8d46ed13b6007685`.

## Routes and RTL

Routes are `/`, `/experiments`, `/experiments/[regime]`, `/datasets/real`,
`/datasets/synthetic`, `/training`, `/evaluation`, `/analysis`, `/inference`, `/reports`,
`/reproducibility`, `/system`, and `/about`.

The document is `lang="ar" dir="rtl"`; Tajawal is optimized by Next.js. Technical values use
explicit LTR isolation. The shell includes keyboard navigation, focus states, reduced motion,
mobile navigation, labelled controls, and non-colour status indicators.
