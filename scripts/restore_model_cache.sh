#!/usr/bin/env bash
# Restore a verified model-store snapshot from an explicitly mounted Kaggle
# dataset. Never writes to /kaggle/input and never overwrites a live model store.
set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${FORGENOVAX_WORK_DIR:-/kaggle/working}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$work_dir/ollama-models}"
input="${FORGENOVAX_MODEL_CACHE_INPUT:-}"

if [[ -z "$input" ]]; then
  echo "Model cache: no FORGENOVAX_MODEL_CACHE_INPUT; using existing cache or downloading"
  exit 0
fi
if [[ ! "$input" = /kaggle/input/* ]]; then
  echo "FORGENOVAX_MODEL_CACHE_INPUT must be a mounted path beneath /kaggle/input/" >&2
  exit 64
fi
if [[ ! -d "$input" ]]; then
  echo "Mounted model-cache directory not found: $input" >&2
  exit 1
fi
if [[ -e "$OLLAMA_MODELS" ]]; then
  echo "Model cache: existing store at $OLLAMA_MODELS; not overwriting"
  exit 0
fi
cd "$repo_dir"
python3 -m src.model_cache restore --input "$input" --target "$OLLAMA_MODELS"
