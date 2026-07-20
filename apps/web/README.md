# SynthDet Arabic dashboard and inference laboratory

The Arabic-first Next.js dashboard presents the five verified training runs, sealed Sprint 5 test
campaign, final ranking, per-class/object-size/latency results, deterministic error metadata, and
FastAPI-backed inference laboratory. Repository mode consumes only the generated metadata snapshot
in `src/data/generated/project-snapshot.json`; protected test pixels and model weights are never
placed in frontend assets.

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
- `NEXT_PUBLIC_DATA_MODE=api`: fetches and identity-checks the running FastAPI service.
- `NEXT_PUBLIC_APP_URL`: canonical URL for metadata.
- `NEXT_PUBLIC_API_URL`: browser-facing FastAPI base URL, normally `http://localhost:8000`.
- `SYNTHDET_API_URL`: optional server-side FastAPI base URL for API data mode.

API mode never falls back to repository or demo values. Repository mode exposes sealed final
metrics but never raw dataset paths, protected pixels, or weights. Demo mode remains visibly
labelled and cannot be mistaken for final results.

## Snapshot regeneration

From the repository root run `python scripts/export_dashboard_snapshot.py`. The exporter derives the
dashboard payload from tracked frozen manifests, verified training intake, sealed evaluation and
analysis reports, and hash records. `npm run snapshot:validate` requires all five completed regimes,
the preregistered `real_only` winner, five result hashes, ten prediction hashes, the single
authorized campaign, zero training-time test access, and absence of protected content.

## Inference laboratory

Start FastAPI on port 8000, then run the frontend. Uploads go directly to
`POST /api/v1/inference`; failures remain explicit and never display demo detections. The recommended
model is selected by default, all five models remain selectable, thresholds are bounded, and the UI
shows original/annotated images, detections, timing, JSON, and an annotated-result download. Known
protected-test content hashes are rejected by the API.

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
