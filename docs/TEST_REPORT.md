# Test report

Report date: 2026-10-25

## Environment-independent checks

Run before delivery:

```bash
python3 -m pytest -q
bash -n scripts/*.sh
python3 -m json.tool notebooks/forgenovax_kaggle_gpu_worker.ipynb >/dev/null
```

Coverage includes authentication, OpenAI passthrough, structured-output
validation, request/response limits, timeouts/service interruption, readiness,
GPU diagnostics, cache corruption/reference integrity, non-destructive restore,
model configuration, benchmark calculations, research scoring, and
measurement-gated recommendations.

Local result after implementation: **46 passed**. The only emitted warning was
a third-party Starlette test-client deprecation notice; it did not affect the
worker runtime or test outcomes. Shell syntax, notebook JSON/Python syntax, and
Python byte-compilation also passed.

## Live acceptance

The existing Kaggle account and notebook were accessed on 2026-10-25. The
deployed checkout reached commit `67b87fe`; the final Kaggle test run reported
**45 passed in 2.30s**.

Live acceptance evidence:

- T4 x2, CUDA/PyTorch compatibility, CPU, RAM, disk, and clean initial process
  state were measured.
- Missing authentication returned 401; authenticated models and inference
  readiness returned 200.
- qwen3:14b, gpt-oss:20b, and qwen3-coder:30b completed 4K/8K performance,
  schema-quality, placement, and concurrency trials.
- The qwen3:14b cache exported 8 files / 9,276,205,985 bytes, restored into a
  new model-store path, and passed real gateway inference from that path.
- The recommended qwen3-coder:30b configuration passed local health, public
  health, authenticated public model discovery, and public inference readiness.
- `scripts/status.sh` ended with `OVERALL STATUS: READY`.
- Managed processes and the Kaggle session were stopped after collection to
  prevent further GPU-quota use.

No unit test or mocked response is counted as live GPU acceptance.

## Live session 2 — 2026-10-25 (LinkedIn Intelligence integration)

Run in the existing notebook `forgenovaxinc/forgenovax-kaggle-gpu-worker`
(Accelerator GPU T4 x2, Internet on, secret `FORGENOVAX_API_KEY` attached). The
hosted draft was still the 2026-09-22 notebook (32K context, no
`OLLAMA_SCHED_SPREAD`, cloned `main`), so this branch's notebook was imported
and the worker source pinned to `feat/standalone-reliability-model-cache`
(`bc1632b`). Saved Version 2 remains in the notebook's version history.

| Check | Result |
|---|---|
| GPUs | 2 × Tesla T4, 15,360 MiB each, driver 580.178.04 |
| In-notebook test suite | passed (Run All continued past the pytest cell) |
| `scripts/status.sh` | `OVERALL STATUS: READY` |
| Model placement (`ollama ps`) | `qwen3-coder:30b`, 19 GB, **100% GPU**, context 8192; GPU0 9,841 MiB, GPU1 9,311 MiB (split across both) |
| Disk after model load | 18 GB used of 20 GB, 2.2 GB free (`/kaggle/working/ollama-models` 18 GB) |
| Listening sockets | `127.0.0.1:8000` (gateway) and `127.0.0.1:11434` (Ollama) only |
| External `/healthz` (from a Mac, through the Quick Tunnel) | 200 |
| External `/v1/models`, `/readyz` without key | 401, 401 |
| External invalid key / previous key | 401 |
| External Ollama path `/api/tags` | 404 (not exposed) |
| External authenticated `/v1/models` | 200, `["qwen3-coder:30b"]` |
| External authenticated `/readyz` | 200, `ready: true`, `inference: verified`, 0.43 s |
| External structured completion (JSON Schema, synthetic post) | 200, schema-valid; 47.9 s first (model load after idle), **1.11 s** warm; usage reported (55 prompt / 30–33 completion tokens) |
| Gateway key rotation | new secret verified by fingerprint in the kernel; only uvicorn restarted; tunnel URL unchanged; previous key then rejected (401) |
| LinkedIn Intelligence client (`KaggleLlm` + `PublicResearchRouter`) | real `post_triage` call: schema-valid, 4.0 s, usage recorded as `kaggle/qwen3-coder:30b` (666 in / 167 out); a `draft_message` task was not sent to Kaggle |
| Session stopped afterwards | yes (GPU quota) |

The Quick Tunnel hostname is not recorded: it expired with the session.

## Live session 3 — 2026-10-25 (readiness reload fix, hybrid routing)

The notebook from this branch was imported again (data sources and secret kept; the
Secrets panel did not list the secret, but the kernel retrieved it and its fingerprint
matched the client copy). Run All installed and downloaded the model in ≈ 6 min.
Cell 9 failed once inside Kaggle with `ConnectError: [Errno 101] Network is
unreachable` even though `start_tunnel.sh` had already verified public inference; the
cell now retries transient transport errors.

| Check | Result |
|---|---|
| Model placement | `qwen3-coder:30b`, 19 GB, 100% GPU, 8192; GPU0 9,853 MiB, GPU1 9,323 MiB; disk 2.2 GB free |
| External auth | `/healthz` 200; no key / wrong key 401; `/v1/models` 200; `/readyz` verified |
| **Readiness-induced reload (before `4cd8731`)** | `/readyz` after the 30 s cache: **47.17 s**; next 8,192-context request: **43.05 s** (probe used `num_ctx` 2048, so Ollama reloaded the model twice) |
| After `4cd8731` (applied live: fetch + `scripts/restart_proxy.sh`, tunnel unchanged) | `/readyz` **0.43 s**; next request **0.29 s**; warm 0.19–0.48 s |
| LinkedIn Intelligence live hybrid test (owner API + `ctx.llm()`) | 3/3: endpoint verified and saved; public triage 2.9 s and analysis 4.3 s on the worker ($0); failover to Claude Haiku when the session URL was gone |
| Real expired tunnel after stopping the session | Cloudflare 530 → one recorded failed attempt → Claude answered; next call skipped the worker (back-off) |
| `scripts/status.sh` | `OVERALL STATUS: READY`; session stopped afterwards |

