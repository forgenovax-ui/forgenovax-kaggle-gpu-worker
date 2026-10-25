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

## Current recommendation status

No model recommendation is recorded here until the live three-model run
completes. The existing `qwen3-coder:30b` default is preserved meanwhile. This
is intentional: a hardware inventory or unit-test pass is not evidence that a
different model is faster or more accurate on the target workload.

`qwen3-coder:30b` remains an available coding model regardless of the eventual
routine-research recommendation.
