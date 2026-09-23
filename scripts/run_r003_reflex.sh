#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
worker_dir="$(cd -- "$script_dir/.." && pwd)"
work_dir="${FORGENOVAX_WORK_DIR:-/kaggle/working}"
source_dir="${FNX_R003_SOURCE_DIR:-$work_dir/forgenovax-fnx-r003}"
source_repo="${FNX_R003_REPO_URL:-https://github.com/forgenovax-ui/forgenovax-fnx.git}"
source_sha="${FNX_R003_SOURCE_GIT_SHA:?FNX_R003_SOURCE_GIT_SHA is required}"
artifact_root="${FNX_R003_ARTIFACT_ROOT:-$work_dir/fnx-artifacts/FNX-R003}"
stage="${FNX_R003_STAGE:-pipeline}"

if [[ ! "$source_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FNX_R003_SOURCE_GIT_SHA must be an exact lowercase 40-character SHA" >&2
  exit 64
fi
if [[ "$stage" != "pipeline" && "$stage" != "full" ]]; then
  echo "FNX_R003_STAGE must be pipeline or full" >&2
  exit 64
fi

gpu_inventory() {
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version \
    --format=csv,noheader,nounits
}

require_idle_t4_x2() {
  local inventory count used index name total utilization driver
  inventory="$(gpu_inventory)"
  count="$(printf '%s\n' "$inventory" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [[ "$count" != "2" ]] || [[ "$(printf '%s\n' "$inventory" | grep -c 'T4')" != "2" ]]; then
    echo "FNX-R003 requires exactly two visible NVIDIA T4 GPUs" >&2
    exit 1
  fi
  while IFS=',' read -r index name used total utilization driver; do
    used="${used// /}"
    if (( used > 512 )); then
      echo "GPU $index retains ${used} MiB after inference shutdown" >&2
      exit 1
    fi
  done <<< "$inventory"
  printf '%s\n' "$inventory"
}

prepare_source() {
  if [[ ! -d "$source_dir/.git" ]]; then
    git clone --filter=blob:none "$source_repo" "$source_dir"
  fi
  git -C "$source_dir" fetch --depth 1 origin "$source_sha"
  git -C "$source_dir" checkout --detach "$source_sha"
  test "$(git -C "$source_dir" rev-parse HEAD)" = "$source_sha"
  python -m pip install -q -e "${source_dir}[research,dev]"
  python - <<'PY'
import peft
import torch
import transformers

if tuple(map(int, transformers.__version__.split(".")[:2])) < (5, 13):
    raise SystemExit(f"Transformers lacks required native Qwen3.5 support: {transformers.__version__}")
print({"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__})
PY
  cd "$source_dir"
  python scripts/audit_r003_leakage.py
  python -m pytest -q
}

start_gpu_sampler() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  (
    while true; do
      printf '{"sampled_at":"%s","gpus":' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits | python -c '
import json, sys
rows=[]
for line in sys.stdin:
    values=[item.strip() for item in line.split(",")]
    rows.append({"index":int(values[0]),"name":values[1],"utilization_percent":float(values[2]),"memory_used_mib":float(values[3]),"memory_total_mib":float(values[4]),"power_watts":float(values[5])})
print(json.dumps(rows, separators=(",",":")) + "}")'
      sleep 5
    done
  ) >> "$output" &
  sampler_pid="$!"
}

stop_gpu_sampler() {
  if [[ -n "${sampler_pid:-}" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}

run_candidate() {
  local name="$1" config="$2" rows="$3"
  local output="$artifact_root/$name"
  mkdir -p "$output"
  start_gpu_sampler "$output/gpu-samples.jsonl"
  trap stop_gpu_sampler RETURN
  CUDA_VISIBLE_DEVICES=0,1 HF_HOME="$work_dir/huggingface-r003" \
    python "$source_dir/scripts/train_reflex_r003.py" \
      --config "$source_dir/$config" \
      --output "$output" \
      --max-train-rows "$rows" \
      --max-validation-rows 512 \
      --source-revision "$source_sha" \
      --resume
  stop_gpu_sampler
  trap - RETURN
}

select_candidate() {
  python - "$artifact_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
scores = {}
for name in ("FNX-R003-HEADS-001", "FNX-R003-LORA-001"):
    manifest = json.loads((root / name / "run-manifest.json").read_text(encoding="utf-8"))
    scores[name] = manifest["history"][-1]["validation"]["loss"]
selected = min(scores, key=scores.get)
payload = {
    "selection_metric": "validation_nll_surrogate_loss",
    "scores": scores,
    "selected": selected,
    "held_out_accessed": False,
    "production_status": "NOT_PRODUCTION_CERTIFIED",
}
(root / "configuration-selection.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(selected)
PY
}

persist_private_output() {
  local archive="$work_dir/FNX-R003-private-artifacts.tar.gz"
  local manifest="$work_dir/FNX-R003-private-artifacts.manifest.json"
  tar -czf "$archive" -C "$artifact_root" .
  python - "$archive" "$manifest" "$source_sha" "$worker_dir" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

archive, manifest = map(Path, sys.argv[1:3])
source_sha, worker_dir = sys.argv[3:]
digest = hashlib.sha256()
with archive.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
worker_sha = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=worker_dir, check=True, capture_output=True, text=True
).stdout.strip()
payload = {
    "experiment_id": "FNX-R003",
    "archive": archive.name,
    "archive_bytes": archive.stat().st_size,
    "archive_sha256": digest.hexdigest(),
    "source_git_sha": source_sha,
    "worker_git_sha": worker_sha,
    "private_destination": "Kaggle notebook output",
    "production_status": "NOT_PRODUCTION_CERTIFIED",
    "created_at": datetime.now(UTC).isoformat(),
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
}

"$script_dir/shutdown.sh"
require_idle_t4_x2
prepare_source
mkdir -p "$artifact_root"

if [[ "$stage" == "pipeline" ]]; then
  run_candidate "FNX-R003-HEADS-001" "configs/r003/fnx_reflex_2b_heads.yaml" 1024
  run_candidate "FNX-R003-LORA-001" "configs/r003/fnx_reflex_2b_lora.yaml" 1024
  select_candidate
else
  selected="$(select_candidate)"
  if [[ "$selected" == "FNX-R003-LORA-001" ]]; then
    config="configs/r003/fnx_reflex_2b_lora.yaml"
  else
    config="configs/r003/fnx_reflex_2b_heads.yaml"
  fi
  run_candidate "FNX-R003-FULL-001" "$config" 28000
fi

persist_private_output
