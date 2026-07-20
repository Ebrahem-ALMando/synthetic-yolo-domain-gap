# SynthDet Finalization Plan

This checklist is the durable execution plan for completing SynthDet. Scientific gates are ordered
so the protected real test set is never opened before the Sprint 5 contract is frozen, committed,
and pushed. A failed independent task does not stop work that cannot affect that gate.

## Phase 0 — Recovery and durable state

- [x] Read `AGENTS.md`, `PROJECT_STATE.md`, `README.md`, and the durable goal attachment.
- [x] Inspect branch, HEAD, origin, upstream synchronization, Git status, staged/untracked files,
  ignored artifacts, dashboard implementation, returned artifacts, and available reports.
- [x] Confirm the tracked worktree is clean and no dataset, checkpoint, archive, secret, or
  protected-test image is staged.
- [x] Create `FINALIZATION_PLAN.md` and `FINALIZATION_STATE.json`.
- [x] Identify and stop only stale project-owned processes if they interfere with validation.

## Phase 1 — Sprint 6A dashboard foundation

- [x] Audit all required routes, RTL shell, Tajawal, themes, navigation, charts, tables, states,
  accessibility, responsive behavior, technical LTR isolation, and official logo integrity.
- [x] Validate widths 1440, 1280, 1024, 768, and 390 pixels with ignored screenshots.
- [x] Run snapshot validation, TypeScript, ESLint, unit tests, production build, route/browser smoke,
  accessibility/console checks, Python tests, Ruff, and `git diff --check`.
- [x] Fix only verified defects and commit/push the completed Sprint 6A gate (`ea719be`).

## Phase 2 — Sprint 4B CUDA artifact intake

- [x] Validate sidecar, archive SHA-256, external inventory schema, paths, sizes, hashes, duplicates,
  symlinks, forbidden content, and identity fields without touching protected-test pixels.
- [x] Extract once to ignored `artifacts/external_training/sprint4b-v2/extracted/` using safe paths.
- [x] Validate exactly five completed regimes, shared frozen profile/configuration, zero training test
  access, result CSV integrity, checkpoint hashes, loadability, YOLO11n architecture, and class order.
- [x] Generate tracked validation-only intake reports clearly marked non-final.
- [ ] Update factual project state/documentation; validate, commit, and push without weights/archive.
  (Documentation updated; validation/commit/push pending.)

## Phase 3 — Freeze Sprint 5 evaluation contract

- [ ] Derive one shared evaluation configuration from frozen repository protocols and installed
  environment; record manifest/checkpoint hashes, policies, schema, ranking, latency, and campaign ID.
- [ ] Create and validate `configs/evaluation/sprint5_final.yaml`, input contract JSON, evaluator,
  and `docs/evaluation_protocol.md` without reading test metrics or image pixels.
- [ ] Commit and push the contract; verify local HEAD equals `origin/main`.

## Phase 4 — One locked protected-test campaign

- [ ] Verify the committed contract, five checkpoint hashes, fixed 68-image manifest, image/label
  integrity, split leakage, class order, and clean revision before authorized access.
- [ ] Create the campaign lock and execute one complete comparable five-model campaign.
- [ ] On infrastructure failure only, preserve the attempt log and rerun all five unchanged under a
  new attempt ID; never tune or inspect partial metrics for decisions.
- [ ] Seal predictions/results and generate complete metrics, ranking, curves, plots, hashes,
  environment evidence, and test-access audit.

## Phase 5 — Final scientific analysis

- [ ] Select the recommendation using only the preregistered ranking rule.
- [ ] Generate domain-gap, marginal data-efficiency, generalization, per-class, object-size, latency,
  confusion, limitation, and deterministic error-analysis reports.
- [ ] Keep all galleries containing protected images ignored; track metadata only.

## Phase 6 — FastAPI inference backend

- [ ] Implement typed lazy model registry and all required project/evaluation/training/report/model
  endpoints plus bounded inference with CPU/CUDA selection and structured errors.
- [ ] Enforce upload/MIME/decode/bounds/time limits, safe names, cleanup, no arbitrary paths, no
  private absolute paths, and protected-test content-hash rejection.
- [ ] Add environment example, OpenAPI descriptions, README, unit/integration/security/registry and
  non-test inference smoke tests.

## Phase 7 — Verified dashboard integration

- [ ] Export verified repository metadata and implement strict repository/demo/API modes with no
  silent fallback or unlabelled fabricated final values.
- [ ] Integrate training, evaluation, ranking, class/size/latency, analysis, identities, hashes,
  campaign audit, recommendation, reports, and model availability.
- [ ] Connect the inference laboratory to FastAPI with upload, controls, overlay, table, JSON,
  annotated output, download, and complete states.

## Phase 8 — Product polish

- [ ] Recheck every route in light/dark modes at desktop/laptop/tablet/mobile widths.
- [ ] Resolve RTL, overflow, hashes, chart/table/dialog/navigation/print, loading/error/empty,
  reduced-motion, console, hydration, dead-control, broken-link, and presentation-quality defects.

## Phase 9 — Academic reports

- [ ] Produce the required Arabic/English reports, summaries, methodology, results, limitations,
  reproducibility appendix, demo guide, defense questions, numbered figures, and tables.
- [ ] Use only verified repository references and clearly distinguish validation, final test, demo,
  and interpretation; generate PDFs only when reliable tooling is available.

## Phase 10 — Presentation and defense

- [ ] Generate an editable 14–18-slide source and validated PPTX using the official identity and
  verified figures, plus Arabic notes, demo script, defense Q&A, checklist, and 5/10–15 minute plans.

## Phase 11 — Local demo and release

- [ ] Add frontend/backend Dockerfiles, compose, environment templates, run/stop scripts, health
  checker, safe CORS, model placement diagnostics, and graceful shutdown.
- [ ] Build ignored full local-demo and source-only release packages with inventory, identity,
  instructions, and SHA-256 sidecars; exclude protected test images from both.

## Phase 12 — Acceptance validation

- [ ] Run every applicable scientific, Python, backend, frontend, browser, end-to-end, packaging,
  security, documentation, and portability gate.
- [ ] Record exact commands, outcomes, evidence, and honest limitations in
  `FINAL_ACCEPTANCE_REPORT.md`; never mark a failed or unavailable gate passed.

## Phase 13 — Git review and delivery

- [ ] Before each logical commit inspect status/diff, run whitespace/secret/forbidden-artifact
  checks, and ensure no weights, archives, datasets, protected images, caches, or unrelated work.
- [ ] Create normal commits and push normally to `origin/main`; never force, amend, rewrite, or
  change remotes.
- [ ] Finish with `main` tracking `origin/main`, ahead/behind 0/0, clean tracked worktree, and local
  ignored scientific/release artifacts preserved.

## Phase 14 — Final project state

- [ ] Update root/app READMEs, `PROJECT_STATE.md`, decision/reproducibility/training/evaluation/Sprint
  6 documents, final findings, commands, links, limitations, and repository structure.
- [ ] Set `project_complete_submission_ready` only when every mandatory scientific, application,
  documentation, acceptance, release, and synchronization gate passes; otherwise use the precise
  reduced status and record blockers.
