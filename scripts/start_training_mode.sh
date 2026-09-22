#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work_dir="${FORGENOVAX_WORK_DIR:-/kaggle/working}"
fnx_dir="${FNX_SOURCE_DIR:-$work_dir/forgenovax-fnx}"
fnx_repo_url="${FNX_REPO_URL:-}"
fnx_git_sha="${FNX_GIT_SHA:-}"
experiment_id="${FNX_EXPERIMENT_ID:-FNX-R001}"
hf_home="${HF_HOME:-$work_dir/huggingface}"

if [[ ! "$experiment_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "Unsafe FNX_EXPERIMENT_ID" >&2
  exit 64
fi

"$script_dir/shutdown.sh"
inference_processes_running() {
  pgrep -f '[o]llama|[u]vicorn.*src\.proxy|[c]loudflared.*127\.0\.0\.1' >/dev/null
}
for _ in {1..30}; do
  if ! inference_processes_running; then
    break
  fi
  sleep 1
done
if inference_processes_running; then
  echo "Inference processes did not stop cleanly" >&2
  exit 1
fi

gpu_inventory="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader,nounits)"
gpu_count="$(printf '%s\n' "$gpu_inventory" | sed '/^$/d' | wc -l | tr -d ' ')"
if [[ "$gpu_count" != "2" ]] || [[ "$(printf '%s\n' "$gpu_inventory" | grep -c 'T4')" != "2" ]]; then
  echo "Training mode requires exactly two visible NVIDIA T4 GPUs" >&2
  exit 1
fi
while IFS=',' read -r index name used total driver; do
  used="${used// /}"
  if (( used > 512 )); then
    echo "GPU $index still has ${used} MiB allocated after inference shutdown" >&2
    exit 1
  fi
done <<< "$gpu_inventory"

export HF_HOME="$hf_home"
export TRANSFORMERS_CACHE="$hf_home/transformers"
mkdir -p "$HF_HOME" "$work_dir/fnx-artifacts/$experiment_id"

if [[ ! -d "$fnx_dir/.git" ]]; then
  if [[ -z "$fnx_repo_url" ]]; then
    echo "PERSISTENCE_BLOCKED: provide an authorized FNX_SOURCE_DIR or FNX_REPO_URL" >&2
    exit 78
  fi
  git clone --filter=blob:none "$fnx_repo_url" "$fnx_dir"
fi
if [[ -n "$fnx_git_sha" ]]; then
  git -C "$fnx_dir" fetch --depth 1 origin "$fnx_git_sha"
  git -C "$fnx_dir" checkout --detach "$fnx_git_sha"
fi
resolved_sha="$(git -C "$fnx_dir" rev-parse HEAD)"
if [[ -n "$fnx_git_sha" ]] && [[ "$resolved_sha" != "$fnx_git_sha" ]]; then
  echo "FNX checkout does not match requested revision" >&2
  exit 1
fi

cd "$fnx_dir"
python -c 'import json, platform; print(json.dumps({"python": platform.python_version()}))'
python scripts/validate_data.py
python scripts/train.py \
  --experiment "$experiment_id" \
  --output "$work_dir/fnx-artifacts/$experiment_id" \
  --resume
