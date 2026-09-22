# ForgeNovaX Kaggle GPU Worker

## Overview

This project turns an ephemeral Kaggle **GPU T4 x2** notebook session into an authenticated, OpenAI-compatible development inference worker. It is intended for ForgeNovaX engineering, QA, evaluation, batch inference, and experiments—not permanent production traffic.

The worker now has mutually exclusive `MODE=inference` and `MODE=training` dispatch. Existing inference behavior is preserved; training mode releases and verifies both GPUs before launching an exact FNX revision. See [worker modes](docs/MODES.md).

The default model is `qwen3-coder:30b`. Only the selected model is downloaded.

## Architecture

```text
Approved OpenAI-compatible client
                │ HTTPS + bearer token
                ▼
       Cloudflare Quick Tunnel
                │ loopback HTTP
                ▼
 Authenticated FastAPI gateway :8000
                │ token removed
                ▼
       Ollama (loopback) :11434
                │
                ▼
       Kaggle GPU 0 + GPU 1
```

Ollama is never the tunnel target. The public path is always Cloudflare → authenticated proxy → Ollama.

## Requirements

- Kaggle account with GPU access (phone verification may be required)
- Notebook accelerator set to **GPU T4 x2**
- Notebook Internet access enabled
- A Kaggle secret named `FORGENOVAX_API_KEY` with a long random value (recommended)

Kaggle supplies two distinct 16 GB T4 devices; it is not one 32 GB GPU. Model placement is reported only from actual runtime evidence.

## Quick Start

1. Follow [Kaggle setup](docs/KAGGLE_SETUP.md).
2. Run every notebook cell and wait for `STATUS: READY`.
3. Copy the displayed OpenAI Base URL.
4. Configure Hermes or another OpenAI-compatible client with the URL, API key, and model.

For a non-notebook shell inside Kaggle:

```bash
./scripts/install.sh
./scripts/start_ollama.sh
ollama pull "${MODEL:-qwen3-coder:30b}"
./scripts/start_proxy.sh
./scripts/start_tunnel.sh
./scripts/status.sh
```

## Kaggle Setup

The exact non-engineer workflow, secret setup, and expected output are in [docs/KAGGLE_SETUP.md](docs/KAGGLE_SETUP.md). The notebook performs GPU detection, installation, readiness polling, model tests, authentication tests, listener isolation, tunnel startup, and full public inference validation.

## Starting Worker

Use [notebooks/forgenovax_kaggle_gpu_worker.ipynb](notebooks/forgenovax_kaggle_gpu_worker.ipynb). Startup logs are written beneath `/kaggle/working`; no arbitrary sleep is used as a readiness signal.

The configured defaults are:

```text
MODEL=qwen3-coder:30b
CONTEXT_LENGTH=32768
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=20m
OLLAMA_HOST=127.0.0.1:11434
PROXY_HOST=127.0.0.1
PROXY_PORT=8000
```

## Connecting Hermes

See [docs/HERMES_SETUP.md](docs/HERMES_SETUP.md) for the exact provider fields and verification prompt.

## Connecting Other Clients

Set `FORGENOVAX_BASE_URL` to the displayed URL ending in `/v1`.

### curl

```bash
curl "$FORGENOVAX_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $FORGENOVAX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder:30b",
    "messages": [
      {
        "role": "user",
        "content": "Reply only with FORGENOVAX_CLIENT_READY"
      }
    ],
    "stream": false
  }'
```

### Python (OpenAI SDK)

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["FORGENOVAX_BASE_URL"],
    api_key=os.environ["FORGENOVAX_API_KEY"],
)
response = client.chat.completions.create(
    model="qwen3-coder:30b",
    messages=[
        {"role": "user", "content": "Reply only with FORGENOVAX_CLIENT_READY"}
    ],
    stream=False,
)
print(response.choices[0].message.content)
```

Install the client SDK separately with `pip install openai`; it is not required by the worker.

## Model Selection

Change only `MODEL` before startup. Supported operating choices are:

- `qwen3-coder:30b` (default; strongest coding option, highest resource demand)
- `gpt-oss:20b` (general reasoning/coding alternative)
- `qwen3:14b` (smaller and generally faster)

The worker downloads only the active model. See [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md).

## Health Checks

Run:

```bash
./scripts/status.sh
```

Each line is based on a live check. A missing GPU, unloaded model, stopped process, failed auth gate, or unavailable public endpoint is shown as `FAIL`; the script never converts an unchecked condition to `PASS`.

## Security

Read [docs/SECURITY.md](docs/SECURITY.md) before use. Key properties:

- Ollama and FastAPI bind only to loopback.
- all `/v1/*` routes require a bearer token;
- `/healthz` is public but contains no secrets;
- the Cloudflare tunnel targets port 8000, never port 11434;
- actual credentials are never committed or included in access logs.

## Shutdown

```bash
./scripts/shutdown.sh
```

Then stop the Kaggle session in the Kaggle interface or GPU quota may continue to be consumed. See [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Troubleshooting

Use the notebook diagnostics cell and [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). The three runtime logs are:

```text
/kaggle/working/ollama.log
/kaggle/working/proxy.log
/kaggle/working/cloudflared.log
```

## Limitations

- Kaggle runtimes, storage, URLs, and temporary keys are ephemeral.
- Quick Tunnels provide no uptime or hostname guarantee.
- Cold model loading can make the first request slow.
- The worker is single-operator development compute, not a multi-tenant service.
- Live GPU/model/tunnel acceptance can only be completed inside Kaggle.
- Regulated, confidential production, payment, trading, production authentication, voice, and persistent customer workloads are out of scope.

## Future Compute Router Integration

The stable OpenAI-compatible boundary allows a future ForgeNovaX Compute Router to select among local Mac/oMLX, this Kaggle worker, a ForgeNovaX GPU server, or paid GPU cloud. Routing may consider privacy, cost, latency, GPU/model/context requirements, task type, and availability. That router is intentionally not implemented here.
