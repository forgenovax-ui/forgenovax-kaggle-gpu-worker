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

Local result after implementation: **43 passed**. The only emitted warning was
a third-party Starlette test-client deprecation notice; it did not affect the
worker runtime or test outcomes. Shell syntax, notebook JSON/Python syntax, and
Python byte-compilation also passed.

## Live acceptance

The existing Kaggle account and notebook were accessed on 2026-10-25. T4 x2,
Internet, CUDA/PyTorch compatibility, CPU, RAM, disk, and initial process state
were measured. Final rows for Ollama inference, external tunnel, three-model
comparison, and restored-cache inference must be filled from the live session
before the PR is marked ready. No unit test or mocked response is counted as
live GPU acceptance.
