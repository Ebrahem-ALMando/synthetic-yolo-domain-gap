# Sprint 6 dashboard plan

## Sprint 6A — dashboard foundation

Status: `sprint6a_dashboard_foundation_implemented`.

- Isolated strict Next.js/TypeScript application with Tailwind and Radix primitives.
- Arabic RTL shell, light/dark modes, mobile navigation, and command palette.
- All required overview, experiment, dataset, training, evaluation, analysis, inference, report,
  reproducibility, system, and about routes.
- Typed repository/demo/API architecture and deterministic safe snapshot export.
- Locked final-result states and unavoidable demo labelling.
- Responsive CSS, accessible controls, Recharts, TanStack Table, tests, and production build gate.

The brand integration is complete and uses the exact supplied bitmap without redrawing or
recompression.

## Sprint 6B — remaining integration

- Ingest and validate returned Sprint 4B result archives.
- Add authenticated read-only API and storage adapters.
- Connect completion state, checkpoints, evaluation artifacts, and model registry.
- Implement approved inference without bundling weights in the browser.
- Replace marked mock cases with traceable Sprint 5 failure artifacts.
- Add Playwright/axe browser regression and visual baselines.
- Configure deployment, observability, and university/team metadata.

Sprint 6 remains incomplete until those integrations and the final scientific protocol finish.
