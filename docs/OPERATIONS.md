# Operations

## Startup order

1. Validate GPU 0 and GPU 1.
2. Install Python dependencies, Ollama, and cloudflared.
3. Start Ollama and poll `/api/tags` until ready.
4. Pull and verify the configured model.
5. Run local inference and capture `ollama ps` plus `nvidia-smi` evidence.
6. Load or generate the API key.
7. Start the FastAPI gateway and run local security tests.
8. Verify the Ollama listener and start a tunnel to port 8000.
9. Run full public acceptance tests.

The notebook automates this sequence and stops on the first failed gate.

## Status

```bash
./scripts/status.sh
```

This checks the NVIDIA runtime and both GPU indices, Ollama API, model installation/loading, proxy, authentication rejection, authorized model listing, cloudflared process, public health, and the authenticated public API. Only all-live checks produce `OVERALL STATUS: READY`.

For the heldout-safe FNX-R002 broadcast preflight, use
`scripts/run_r002_live_preflight.sh`. It runs the frozen preflight-only path,
restarts retained services so the proxy receives the authoritative metrics
path, verifies the unauthenticated allowlisted public metrics route, and stops
before creation or acceptance of any heldout boundary.

## Logs

```text
/kaggle/working/ollama.log
/kaggle/working/proxy.log
/kaggle/working/cloudflared.log
```

Use the notebook diagnostics cell for redacted recent output. Uvicorn access logging is disabled to reduce accidental header or query logging.

## Model lifecycle

`OLLAMA_KEEP_ALIVE=20m` retains the model after a request. After it unloads, the next request incurs warm-up latency. `OLLAMA_NUM_PARALLEL=1` limits concurrent generation pressure for the T4 pair.

## Shutdown

Run:

```bash
./scripts/shutdown.sh
```

This stops the cloudflared, uvicorn, and Ollama processes recorded by the worker and removes the temporary URL file. Then stop the Kaggle notebook session from the Kaggle interface; otherwise GPU quota may continue being consumed.

## Runtime recovery

If a component fails, inspect its log, run shutdown, and rerun notebook cells from installation/startup. After a Kaggle reset, run the entire notebook because PIDs, keys, model files, and URLs may no longer exist.
