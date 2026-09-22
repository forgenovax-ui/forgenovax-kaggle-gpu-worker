#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_dir"

"$script_dir/install.sh"
"$script_dir/start_ollama.sh"
model="${MODEL:-qwen3-coder:30b}"
ollama pull "$model"
"$script_dir/start_proxy.sh"
"$script_dir/start_tunnel.sh"
"$script_dir/status.sh"

