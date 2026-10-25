#!/usr/bin/env bash
# A local model should not be downloaded again on each notebook restart.
set -Eeuo pipefail
model="${MODEL:-qwen3-coder:30b}"
if [[ "$model" == -* || -z "$model" ]]; then
  echo "Invalid MODEL" >&2
  exit 64
fi
if ollama show "$model" >/dev/null 2>&1; then
  echo "Model already installed: $model (skipping download)"
else
  echo "Model not installed: $model (downloading)"
  ollama pull "$model"
fi
ollama show "$model" >/dev/null || {
  echo "Model validation failed after download: $model" >&2
  exit 1
}
