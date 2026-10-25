#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_dir"

"$script_dir/install.sh"
export OLLAMA_MODELS="${OLLAMA_MODELS:-${FORGENOVAX_WORK_DIR:-/kaggle/working}/ollama-models}"
"$script_dir/restore_model_cache.sh"
"$script_dir/start_ollama.sh"
"$script_dir/ensure_model.sh"
"$script_dir/start_proxy.sh"
"$script_dir/start_tunnel.sh"
"$script_dir/status.sh"

