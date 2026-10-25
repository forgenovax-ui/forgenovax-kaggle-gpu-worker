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
