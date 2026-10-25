# Standalone Kaggle optimization

This upgrade is independent of the LinkedIn Intelligence application. It does
not access LinkedIn, require LinkedIn credentials, change the LinkedIn codebase,
or schedule browser actions.

## Quick start

1. Import the repository's notebook into Kaggle.
2. Select the T4 x2 accelerator, turn on Internet, and add the
   `FORGENOVAX_API_KEY` Kaggle secret.
3. Choose a model in the notebook's configuration cell. The measured T4 x2
   default is `qwen3-coder:30b` at 8,192 context. Re-run the comparison before
   changing it for a materially different workload.
4. Run all cells. The worker confirms the GPUs, creates an authenticated
   OpenAI-compatible endpoint, and prints the temporary URL and model.

## Reuse downloaded model files

Kaggle notebook processes and /kaggle/working files do **not** persist
automatically across arbitrary fresh sessions. A saved notebook output or
private Dataset mounted as /kaggle/input can be reused.

One time, after the model has been installed, run inside the project directory:

```bash
python3 -m src.model_cache export \
  --source /kaggle/working/ollama-models \
  --output /kaggle/working/fnx-cache-export
```

This produces `fnx-cache-export/model-cache.json` and
`fnx-cache-export/ollama-models/{blobs,manifests}`. All copied files are
SHA-256 hashed and checked. Current Ollama `manifests-v2` links are preserved
only when they are relative links to in-store SHA-256 blobs; unsafe links are
rejected. Export **duplicates** model bytes temporarily.
Ensure adequate free disk space, and check notebook output size limits before
attempting to save it. Never include any access keys in the cache.

Save this output as a notebook version or create a **private** dataset from
it, and attach that data to the next notebook. The mounted folder must contain
both `model-cache.json` and `ollama-models/`. Before startup, set:

```python
os.environ["FORGENOVAX_MODEL_CACHE_INPUT"] = "/kaggle/input/YOUR-DATASET"
```

Then run the notebook cells. The optional restore step verifies every file
before copying into a new `/kaggle/working/ollama-models` folder. It never
writes into /kaggle/input and refuses to overwrite an existing model store.
Ollama then skips the network download if its requested model is installed.

To verify a mounted dataset without restoring:

```bash
python3 -m src.model_cache verify --input /kaggle/input/YOUR-DATASET
```

**Important:** A checksum catches accidental corruption but is not a digital
signature. Attach only datasets you own and trust.

## Measure whether optimization actually helps

After Ollama starts and the model is installed:

```bash
python3 -m src.benchmark --model qwen3:14b --context 8192 --runs 3 \
  --cold-start --measure-first-token
```

Or set `FORGENOVAX_RUN_BENCHMARK=1` before the notebook's optional
benchmark cell. Reports are written to
`/kaggle/working/benchmarks/benchmark-*.json`.

The benchmark uses synthetic business discussion text (no private LinkedIn
data). Each report includes actual Ollama duration/token counts, throughput,
first-token latency, wall time, sampled GPU memory/utilization, Ollama CPU use,
and measured model VRAM placement. A GPU name or device count alone is **not**
proof the model is running on both devices.
Confirm actual memory allocation with `ollama ps` and `nvidia-smi`.

For a fair comparison, use the **same prompt**, context length, token limit,
warmup count, and runtime load. Compare at least:

- Model response quality and correct JSON output on known synthetic examples.
- Median end-to-end duration and model-generated tokens per second.
- Model load time, available VRAM, and whether CPU fallback occurred.
- End-to-end response from the authenticated OpenAI-compatible endpoint.

The performance tool deliberately does not claim that model quality is
equivalent: it reports performance only.

On the live 2026-10-25 run, `OLLAMA_SCHED_SPREAD=1` increased qwen3:14b
steady-state throughput from 20.43 to 25.62 tokens/s. Keep the spread setting
enabled on T4 x2 unless a matched benchmark for a future Ollama/model release
shows otherwise.

## Session recovery

- If the notebook restarts, rerun the startup cells.
- Model state is reusable **only if saved as output/dataset and mounted again**.
- The Cloudflare Quick Tunnel URL changes after a new tunnel is created.
- Any external client must update its `/v1` base URL after a tunnel reset.
- A notebook session must not be treated as a persistent database or job queue.
- The endpoint is intended for time-bounded experimentation and public,
  non-sensitive inputs, not account credentials or private messages.

## Tests

```bash
python3 -m pytest -q
bash -n scripts/*.sh
```

The tests use synthetic files and mocked HTTP responses. Passing tests
establish code behavior, not GPU hardware readiness. Run the notebook's
live `STATUS: READY` acceptance inside Kaggle before using the endpoint.
