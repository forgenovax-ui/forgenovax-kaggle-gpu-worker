# Engineering audit

Audit date: 2026-10-25

## Repository and delivery state

- `main` was at `96d2e87` when audited.
- Pull request #1 remained open and draft; it had not been merged.
- `feat/standalone-reliability-model-cache` was cleanly based on that `main` and
  contained 15 commits through `b921474`.
- The latest pre-audit GitHub Actions push and pull-request runs were successful.
- The separate `r002-selective-reflex` and `r003-fnx-reflex-2b` branches contain
  training experiments. They were inspected for scope and left unchanged.

## Existing system map

The notebook is the operator entry point. In inference mode it installs the
declared dependencies, restores an optional verified model cache, starts
loopback-only Ollama, ensures the selected model exists, starts the
loopback-only authenticated FastAPI gateway, and exposes only that gateway
through a Cloudflare Quick Tunnel. `src/proxy/app.py` owns the public trust
boundary. `src/healthcheck.py` owns end-to-end status. Shell scripts own process
lifecycle. `src/model_cache.py` owns portable cache integrity. Training mode
first shuts down inference and then invokes an explicitly supplied FNX source
revision; it remains separate from inference artifacts.

## What was already implemented

- T4 x2 detection in the notebook and health report.
- Mutually exclusive inference and training modes.
- Loopback binding for Ollama and FastAPI.
- Mandatory bearer authentication for `/v1/*`.
- OpenAI-compatible passthrough for `/v1/models` and `/v1/chat/completions`.
- Tunnel target isolation, tunnel reuse, and stale-PID-safe shutdown.
- Model-cache export, SHA-256 verification, non-destructive restore, and
  conditional model download.
- Basic Ollama token-throughput measurement.
- Automated proxy/cache/benchmark tests and GitHub Actions.

## Defects and missing evidence found

1. Public `/healthz` proved only that FastAPI responded, not that a model could
   complete authenticated inference.
2. Request and response bodies were unbounded.
3. JSON-schema response requests were forwarded without independent validation
   of model output.
4. Gateway-generated errors were not consistently OpenAI-shaped.
5. GPU evidence was limited to before/after device snapshots. Per-device
   process memory, in-flight utilization, Ollama VRAM placement, CPU use, and
   CPU-offload ratio were absent.
6. The benchmark used one free-form prompt and could not compare research
   quality, contexts, or concurrent requests across all supported models.
7. Cache checksums did not verify that Ollama manifests referenced present
   blobs.
8. Some idempotency checks accepted HTTP liveness without authenticated model
   readiness.
9. Current Ollama creates safe internal `manifests-v2` symlinks; the original
   blanket symlink rejection made live cache export fail closed.
10. Reasoning models could consume the original readiness and evaluation token
    ceilings before emitting user-visible content.

## Focused improvements

- Added authenticated `/readyz` with a real, deterministic model inference and
  short-lived success caching.
- Added byte limits, sanitized OpenAI-shaped gateway errors, and strict
  JSON/JSON-Schema response validation.
- Added measured GPU/CPU/process/Ollama-placement diagnostics and in-flight
  sampling to the benchmark.
- Added a versioned synthetic research dataset, deterministic scoring,
  sequential/concurrent tests, context comparison, and measurement-gated model
  recommendation.
- Added Ollama manifest-to-blob integrity checks.
- Added a narrow, recorded allow-list for Ollama's internal
  `manifests-v2 -> blobs` links while continuing to reject absolute,
  out-of-store, broken, and unexpected links.
- Hardened GPU visibility and startup/tunnel reuse around authenticated
  inference readiness.
- Raised bounded readiness and quality budgets only far enough for reasoning
  models to emit validated content.

## Live delivery outcome

- qwen3:14b cache export, checksum verification, non-overwriting restore, and
  restored-path inference passed.
- `OLLAMA_SCHED_SPREAD=1` improved qwen3:14b median throughput by 25.4% in the
  matched three-run comparison and is now the default.
- All three supported models completed the 4K/8K comparison with 100% final
  schema success.
- `qwen3-coder:30b` at 8,192 context won the measurement-gated recommendation.
- The final recommended runtime passed public authenticated inference and
  `OVERALL STATUS: READY`; the Kaggle session was then stopped.

No LinkedIn Intelligence code, Anthropic integration, training implementation,
or unrelated service was modified.
