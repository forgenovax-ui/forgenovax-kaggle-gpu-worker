# Model Guide

Set `MODEL` before starting Ollama. The pull and all acceptance requests read this single setting.

| Model | Use | Relative resource profile |
|---|---|---|
| `qwen3-coder:30b` | Default code generation and repository reasoning | Highest of the supported choices |
| `gpt-oss:20b` | General reasoning and coding alternative | Medium/high |
| `qwen3:14b` | Faster, lighter general reasoning | Lowest of the supported choices |

Do not pull every model. Ephemeral storage, download time, and quota are limited. Change one value and rerun model pull/startup validation:

```bash
export MODEL=qwen3:14b
```

Actual fit, speed, GPU placement, and context behavior depend on current Ollama packaging, quantization, Kaggle GPU availability, and runtime memory. Use `ollama ps` and `nvidia-smi` as evidence; do not infer that two visible T4 devices guarantee a specific split.

The measured default context is 8,192 tokens. Large contexts consume additional
memory and can force CPU offload. Retain 32,768 only when a separate workload
proves that the extra context is required and placement remains acceptable. See
[model evaluation](MODEL_EVALUATION.md).

Do not change the default research model from assumptions about parameter
count. `src.research_evaluation` emits a recommendation only after all supported
models pass through the same cases, schemas, contexts, and scoring. The live
2026-10-25 comparison selected `qwen3-coder:30b` at 8,192 context.
