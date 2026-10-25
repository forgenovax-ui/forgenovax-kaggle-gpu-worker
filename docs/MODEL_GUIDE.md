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

The historical default context target is 32,768 tokens. Large contexts consume
additional memory and can force CPU offload. Routine research evaluation must
compare 4,096 and 8,192 first; retain 32,768 only when measured quality requires
it and placement remains acceptable. See [model evaluation](MODEL_EVALUATION.md).

Do not change the default research model from assumptions about parameter
count. `src.research_evaluation` emits a recommendation only after all supported
models pass through identical cases, schemas, contexts, and scoring. The coding
model `qwen3-coder:30b` remains supported regardless of that recommendation.
