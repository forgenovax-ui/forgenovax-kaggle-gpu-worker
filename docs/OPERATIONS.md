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

This checks the NVIDIA runtime and both GPU indices, Ollama API, model
installation/loading, measured VRAM placement and CPU offload, proxy,
authentication rejection, authenticated local inference readiness,
cloudflared, public liveness, and authenticated public inference readiness.
Per-GPU process-memory rows are reported as `UNVERIFIED` unless measured and do
not silently claim both devices are used. Only required live checks produce
`OVERALL STATUS: READY`.

Use `python3 -m src.gpu_diagnostics` for the machine-readable inventory and
placement snapshot. Use `python3 -m src.benchmark --cold-start
--measure-first-token` for sampled in-flight evidence.

## Logs

```text
/kaggle/working/ollama.log
/kaggle/working/proxy.log
/kaggle/working/cloudflared.log
```

Use the notebook diagnostics cell for redacted recent output. Uvicorn access logging is disabled to reduce accidental header or query logging.

## Model lifecycle

`OLLAMA_KEEP_ALIVE=20m` retains the model after a request. After it unloads, the next request incurs warm-up latency. `OLLAMA_NUM_PARALLEL=1` limits concurrent generation pressure for the T4 pair. `OLLAMA_SCHED_SPREAD=1` intentionally distributes a model across both visible T4s; set it to `0` only when reproducing Ollama's default single-GPU placement.

## Shutdown

Run:

```bash
./scripts/shutdown.sh
```

This stops the cloudflared, uvicorn, and Ollama processes recorded by the worker and removes the temporary URL file. Then stop the Kaggle notebook session from the Kaggle interface; otherwise GPU quota may continue being consumed.

## Runtime recovery

If a component fails, inspect its log, run shutdown, and rerun notebook cells from installation/startup. After a Kaggle reset, run the entire notebook because PIDs, keys, model files, and URLs may no longer exist.

## Rotate the gateway key without reloading the model

1. Edit the Kaggle secret `FORGENOVAX_API_KEY` (Add-ons → Secrets → ⋯ → Edit
   secret → paste into **Value** → Save). Paste into the Secrets panel, never into
   a cell or the console.
2. In the running notebook, reload it and restart only the gateway:

   ```python
   from kaggle_secrets import UserSecretsClient
   os.environ["FORGENOVAX_API_KEY"] = UserSecretsClient().get_secret("FORGENOVAX_API_KEY").strip()
   subprocess.run(["bash", "scripts/restart_proxy.sh"], check=True, env=os.environ.copy())
   ```

   To compare with the client copy without revealing it, print
   `hashlib.sha256(key.encode()).hexdigest()[:12]` on both sides.
3. Update the client copy (LinkedIn Intelligence: `secrets/kaggle_api_key`). The
   tunnel URL does not change.

