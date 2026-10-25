# Architecture

## Runtime path

```text
Approved client
  → HTTPS Cloudflare Quick Tunnel
  → 127.0.0.1:8000 FastAPI bearer-auth gateway
  → 127.0.0.1:11434 Ollama OpenAI-compatible API
  → configured model on the visible Kaggle GPU devices
```

The gateway is a deliberately small trust boundary. `/healthz` returns only a
boolean and service name and means **liveness only**. Authenticated `/readyz`
performs a short deterministic inference with the configured model and caches a
successful result briefly. Every route under `/v1/` authenticates first,
removes the public authorization header, applies configured body limits,
forwards the original body to Ollama, and strips hop-by-hop headers.

For non-streaming requests with `response_format`, the gateway parses assistant
content as JSON and validates it against the caller's JSON Schema before
returning success. Model output remains untrusted even when Ollama reports HTTP
200. Streaming structured responses are rejected because the complete object
cannot be validated before success.

## Components

- `src/config.py`: one environment-driven configuration source with loopback invariants.
- `src/proxy/app.py`: FastAPI authentication and OpenAI-compatible reverse proxy.
- `src/healthcheck.py`: evidence-based local and public runtime checks.
- `scripts/`: installation, process lifecycle, security preflight, and status commands.
- `notebooks/`: operator workflow and live acceptance tests.

## GPU semantics

Kaggle T4 x2 exposes GPU 0 (16 GB) and GPU 1 (16 GB). It does not expose a single 32 GB GPU. `CUDA_VISIBLE_DEVICES=0,1` makes both devices available to Ollama, but only `ollama ps` and `nvidia-smi` output from the active runtime are accepted as evidence of actual model placement and memory use.

## Failure behavior

The proxy fails closed when its key is unset. The tunnel refuses to start unless Ollama is proven to be loopback-only and local authentication tests return 401/401/200 for missing, invalid, and correct credentials. Install, model pull, acceptance, or tunnel errors stop notebook execution and show recent relevant logs.

## Future router

```text
                    ┌── Local Mac / oMLX
                    │
Agent ─ Compute Router ── Kaggle Worker
                    │
                    ├── ForgeNovaX GPU Server
                    └── Paid GPU Cloud
```

The future router may use privacy, cost, latency, GPU availability, model requirement, context requirement, task type, and availability. It is not part of this project.
