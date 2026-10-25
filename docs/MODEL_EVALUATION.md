# Reproducible model evaluation

The versioned dataset at `benchmarks/research_cases.json` contains synthetic
professional discussions and predefined expected results for post
classification and discussion analysis. The schemas in
`src/research_contracts.py` are defined independently from model output.

## Run the complete comparison

If disk permits, install all three supported models before the full comparison:

```bash
python3 -m src.research_evaluation \
  --models qwen3:14b gpt-oss:20b qwen3-coder:30b \
  --contexts 4096 8192 \
  --runs 2 \
  --concurrency 2
```

The measured Kaggle working disk may not hold all three models together. In
that case, test and remove them sequentially while appending to one report:

```bash
REPORT=/kaggle/working/benchmarks/research-comparison.json
python3 -m src.research_evaluation --models qwen3:14b \
  --contexts 4096 8192 --report "$REPORT"
# After safely removing only qwen3:14b and installing the next model:
python3 -m src.research_evaluation --models gpt-oss:20b \
  --contexts 4096 8192 --report "$REPORT"
python3 -m src.research_evaluation --models qwen3-coder:30b \
  --contexts 4096 8192 --report "$REPORT"
```

The report validates its context set and replaces only the matching
model-context trial, so interrupted runs can resume without inventing or
duplicating measurements.

The command measures cold load, streaming first-token latency, total latency,
generation tokens/sec, sampled GPU memory/utilization, Ollama CPU use,
VRAM-vs-total placement, structured-output success, usage metadata,
classification quality, extraction accuracy, and concurrent-request behavior.

Malformed or schema-incomplete output scores as failure. Missing models are
reported and are never downloaded implicitly. The recommendation remains
`pending_complete_measurement` until every supported model has measured data.
Among models meeting the 90% structured-output gate, quality ranks first and
latency breaks ties.

## Live results — Kaggle T4 x2, 2026-10-25

Each row used the four versioned cases, strict gateway validation, deterministic
temperature, one performance run, and a two-request concurrency probe.

| Model | Context | Schema success | Overall quality | Classification | Extraction | Median case latency | Generation tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3:14b` | 4,096 | 100% | 0.5625 | 0.625 | 0.500 | 19.895 s | 25.57 |
| `qwen3:14b` | 8,192 | 100% | 0.5625 | 0.625 | 0.500 | 16.651 s | 25.38 |
| `gpt-oss:20b` | 4,096 | 100% | 0.6125 | 0.625 | 0.600 | 18.410 s | 49.75 |
| `gpt-oss:20b` | 8,192 | 100% | 0.6125 | 0.625 | 0.600 | 18.401 s | 49.76 |
| `qwen3-coder:30b` | 4,096 | 100% | 0.7250 | 0.750 | 0.700 | 3.478 s | 60.95 |
| `qwen3-coder:30b` | 8,192 | 100% | 0.7250 | 0.750 | 0.700 | 2.572 s | 60.01 |

Usage metadata was present for every successful case. Both requests succeeded
in every concurrency probe. With `OLLAMA_NUM_PARALLEL=1`, the paired requests
were serialized rather than run simultaneously:

| Model | Context | Two-request wall time | Request latencies |
|---|---:|---:|---|
| `qwen3:14b` | 4,096 | 25.668 s | 13.130 s, 25.666 s |
| `qwen3:14b` | 8,192 | 25.713 s | 13.181 s, 25.712 s |
| `gpt-oss:20b` | 4,096 | 50.632 s | 28.806 s, 50.631 s |
| `gpt-oss:20b` | 8,192 | 50.897 s | 50.897 s, 28.908 s |
| `qwen3-coder:30b` | 4,096 | 4.005 s | 2.061 s, 4.003 s |
| `qwen3-coder:30b` | 8,192 | 3.723 s | 1.893 s, 3.722 s |

The gpt-oss trials initially exposed reasoning-budget exhaustion: 16 readiness
tokens and then 512 quality tokens were consumed in the reasoning channel before
user-visible content. The runtime now uses 128 readiness tokens and a bounded
2,048-token quality ceiling. qwen3:14b had already completed all cases under the
earlier 512-token ceiling, and its performance prompt ended naturally at 60
output tokens under a 128-token cap, so those caps were non-binding for qwen.

## Measured recommendation

Use `qwen3-coder:30b` with `CONTEXT_LENGTH=8192`,
`OLLAMA_SCHED_SPREAD=1`, and `OLLAMA_NUM_PARALLEL=1` as the default for this
T4 x2 worker.

It had the highest schema-gated quality (0.725), the lowest median research-case
latency (2.572 s at 8K), and the highest measured generation throughput among
the tested configurations. Its larger 18 GB model-store footprint leaves only
about 2.2 GiB of Kaggle working-disk headroom, so keep the sequential cache and
model-rotation workflow documented above.
