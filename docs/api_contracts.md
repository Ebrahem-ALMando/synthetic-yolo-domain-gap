# Future dashboard API contracts

API mode is intentionally unavailable in Sprint 6A. It must show an error rather than silently
substitute repository or demo data.

Planned read-only endpoints are `GET /api/v1/project/snapshot`, `/experiments`,
`/experiments/{regime}`, `/training/state`, `/evaluation`, `/failures`, and `/system/health`.
`POST /api/v1/inference` accepts user-uploaded images only with explicit model and threshold data.

Responses must not expose secrets, private absolute paths, weights, or protected-test images.
Evaluation records require revision, split, experiment, training, checkpoint, and result identities.
Inference must reject protected-test identifiers and declare `source: "model" | "demo"`.

Errors use `{ code, message_ar, correlation_id, retryable }`. Unavailable models return
`409 MODEL_UNAVAILABLE`, absent scientific results return `409 RESULTS_PENDING`, and protected
requests return `403 PROTECTED_TEST_POLICY`. No endpoint falls back silently.
