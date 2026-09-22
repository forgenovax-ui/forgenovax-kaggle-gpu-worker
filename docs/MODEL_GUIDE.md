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

The default context target is 32,768 tokens. Large contexts consume additional memory, and a model may enforce a different practical limit. Reduce `CONTEXT_LENGTH` if the active runtime cannot load or serve reliably.
