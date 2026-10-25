# External model-router handoff

This contract is ready for a future Claude/Ollama router without adding router
or LinkedIn application code to this repository.

## Endpoint contract

- Base URL: the session-specific Cloudflare URL ending in `/v1`
- Authentication: `Authorization: Bearer <FORGENOVAX_API_KEY>`
- Model identifiers: `qwen3:14b`, `gpt-oss:20b`, `qwen3-coder:30b`
- Discovery: authenticated `GET /v1/models`
- Inference: authenticated `POST /v1/chat/completions`
- Liveness: public `GET /healthz` (gateway process only)
- Readiness: authenticated `GET /readyz` (real model inference)

The router must treat a 200 from `/healthz` as insufficient. Route work only
after `/readyz` returns `ready: true`, the expected `model`, and
`inference: verified`.

Measured default for Kaggle T4 x2:

```text
MODEL=qwen3-coder:30b
CONTEXT_LENGTH=8192
OLLAMA_SCHED_SPREAD=1
OLLAMA_NUM_PARALLEL=1
```

The 2026-10-25 acceptance session passed public health, authenticated model
discovery, and public inference readiness with this configuration. Its Quick
Tunnel hostname was intentionally not recorded because it expired when the
session was stopped.

## Structured requests

Use `stream: false` with either `response_format.type=json_object` or
`response_format.type=json_schema`. For JSON Schema, include
`response_format.json_schema.schema`. The gateway independently parses and
validates assistant content before returning success. Invalid content returns a
sanitized OpenAI-shaped 502 error.

## Retries and session recovery

- Retry 503/504 only when the caller's operation is idempotent or carries its
  own operation identifier. Do not automatically duplicate side-effecting work.
- Use bounded exponential backoff and recheck `/readyz` after a disconnect.
- A restarted Kaggle session receives a new Quick Tunnel hostname; rediscover
  and update the base URL rather than retrying a stale hostname indefinitely.
- A cached model store and notebook files do not make process state or tunnel
  URLs persistent.
- Never send browser cookies, private messages, account credentials, or other
  sensitive account data to this public-data research worker.

## Connected client: ForgeNovaX LinkedIn Intelligence

LinkedIn Intelligence (`packages/research-engine/src/kaggle.ts`) is the
authoritative application; this worker only answers public-post triage and
analysis requests.

- Key: LinkedIn Intelligence `secrets/kaggle_api_key` holds the same value as the
  notebook secret `FORGENOVAX_API_KEY` (install with
  `infrastructure/deployment/set-secret.sh kaggle_api_key --generate --copy`,
  then paste into the Kaggle secret).
- Endpoint: after each session starts, paste the printed **OPENAI BASE URL** into
  LinkedIn Intelligence → Settings → Kaggle GPU inference → *Verify and connect*
  (or `PUT /api/inference/kaggle`). It is saved only after `/readyz` verified
  inference for the served model; no redeploy is needed.
- Routing: Kaggle first for eligible public tasks when ready; otherwise, or on an
  expired tunnel, timeout, gateway schema rejection or a request larger than the
  8,192-token context, the client uses Claude. An unreachable host is backed off
  (60 s, doubling to 30 min) instead of being called repeatedly.
- Accounting: Kaggle calls are recorded as `kaggle/<model>` at $0; failed attempts
  as `kaggle_error:<reason>` with zero tokens.

