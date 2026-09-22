#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
worker_dir="$(cd -- "$script_dir/.." && pwd)"
work_dir="${FORGENOVAX_WORK_DIR:-/kaggle/working}"
source_sha256="${FNX_R002_SOURCE_SHA256:?FNX_R002_SOURCE_SHA256 is required}"
source_tree_sha256="${FNX_R002_SOURCE_TREE_SHA256:?FNX_R002_SOURCE_TREE_SHA256 is required}"
source_git_sha="${FNX_R002_SOURCE_GIT_SHA:?FNX_R002_SOURCE_GIT_SHA is required}"
source_dir="${FNX_R002_SOURCE_DIR:-$work_dir/fnx-r002-src}"
artifact_dir="${FNX_R002_ARTIFACT_DIR:-$work_dir/fnx-r002-artifacts}"
decider_dir="${FNX_R002_DECIDER_DIR:-$work_dir/fnx-upstream-r002/decider}"
decider_revision="c4daaac28af9fea95d627015cffa2dd5a5926ee6"
model="${MODEL:-qwen3-coder:30b}"
live_boundary_file="${FNX_R002_LIVE_BOUNDARY_FILE:-}"
not_before="${FNX_R002_NOT_BEFORE:-2026-09-22T09:00:00-04:00}"
reuse_preexperiment="${FNX_R002_REUSE_PREEXPERIMENT:-false}"
preflight_only="${FNX_R002_PREFLIGHT_ONLY:-false}"
export FNX_PUBLIC_METRICS_PATH="${FNX_PUBLIC_METRICS_PATH:-$artifact_dir/live/metrics.json}"
export FNX_PUBLIC_METRICS_KILL_SWITCH_PATH="${FNX_PUBLIC_METRICS_KILL_SWITCH_PATH:-$artifact_dir/live/PUBLIC_DISABLED}"

gpu_inventory() {
  nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,driver_version \
    --format=csv,noheader,nounits
}

require_t4_x2() {
  local inventory count
  inventory="$(gpu_inventory)"
  count="$(printf '%s\n' "$inventory" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [[ "$count" != "2" ]] || [[ "$(printf '%s\n' "$inventory" | grep -c 'T4')" != "2" ]]; then
    echo "R002 requires exactly two visible NVIDIA T4 GPUs" >&2
    exit 1
  fi
  printf '%s\n' "$inventory"
}

require_idle_gpus() {
  local inventory used index name total driver
  inventory="$(require_t4_x2)"
  while IFS=',' read -r index name used total driver; do
    used="${used// /}"
    if (( used > 512 )); then
      echo "GPU $index still has ${used} MiB allocated" >&2
      exit 1
    fi
  done <<< "$inventory"
}

locate_source_archive() {
  local candidate actual
  while IFS= read -r candidate; do
    actual="$(sha256sum "$candidate" | awk '{print $1}')"
    if [[ "$actual" == "$source_sha256" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(find /kaggle/input -type f -name 'forgenovax-fnx-r002-*.tar.gz' -print | sort)
  echo "No attached R002 source archive matches the required SHA-256" >&2
  return 1
}

source_tree_digest() {
  python - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and candidate.name != "pax_global_header"):
    data = path.read_bytes()
    relative = path.relative_to(root).as_posix()
    rows.append(f"{relative}\0{len(data)}\0{hashlib.sha256(data).hexdigest()}\n")
print(hashlib.sha256("".join(rows).encode()).hexdigest())
PY
}

locate_source_tree() {
  local marker candidate actual
  while IFS= read -r marker; do
    candidate="$(cd -- "$(dirname -- "$marker")/.." && pwd)"
    actual="$(source_tree_digest "$candidate")"
    if [[ "$actual" == "$source_tree_sha256" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(find /kaggle/input -type f -path '*/scripts/run_r002_experiment.py' -print | sort)
  echo "No attached extracted R002 source tree matches the required SHA-256" >&2
  return 1
}

prepare_source() {
  local archive mounted_source source_location
  archive="$(locate_source_archive 2>/dev/null || true)"
  if [[ ! -f "$source_dir/scripts/run_r002_experiment.py" ]]; then
    mkdir -p "$source_dir"
    if [[ -n "$archive" ]]; then
      tar -xzf "$archive" -C "$source_dir"
      source_location="$archive"
    else
      mounted_source="$(locate_source_tree)"
      cp -a "$mounted_source/." "$source_dir/"
      rm -f "$source_dir/pax_global_header"
      source_location="$mounted_source"
    fi
  else
    source_location="$source_dir"
  fi
  test "$(source_tree_digest "$source_dir")" = "$source_tree_sha256"
  printf '%s  %s\n' "$source_tree_sha256" "$source_location"
  python -m pip install -q -e "${source_dir}[research]"
  python -m pip install -q 'transformers==5.17.0' 'torchao==0.16.0'
  python - <<'PY'
import transformers

if transformers.__version__ != "5.17.0":
    raise SystemExit(f"Transformers pin failed: {transformers.__version__}")
print({"transformers": transformers.__version__})
PY
  python "$source_dir/scripts/validate_r002_data.py"
  python "$source_dir/scripts/verify_live_readiness.py"
  python -m pytest -q "$source_dir/tests"
  mkdir -p "$artifact_dir"
  python - "$artifact_dir/source-provenance.json" "$source_git_sha" "$source_sha256" "$source_tree_sha256" "$source_location" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

path, git_sha, archive_sha, tree_sha, source_path = sys.argv[1:]
Path(path).write_text(json.dumps({
    "experiment_id": "FNX-R002",
    "source_git_sha": git_sha,
    "source_archive_sha256": archive_sha,
    "source_tree_sha256": tree_sha,
    "source_location_name": Path(source_path).name,
    "verified_at": datetime.now(UTC).isoformat(),
    "production_status": "NOT_PRODUCTION_CERTIFIED",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

prepare_decider() {
  mkdir -p "$(dirname "$decider_dir")"
  if [[ ! -d "$decider_dir/.git" ]]; then
    git clone --filter=blob:none https://github.com/Mapika/decider "$decider_dir"
  fi
  git -C "$decider_dir" fetch --depth 1 origin "$decider_revision"
  git -C "$decider_dir" checkout --detach "$decider_revision"
  test "$(git -C "$decider_dir" rev-parse HEAD)" = "$decider_revision"
}

run_reflex_pre() {
  "$script_dir/shutdown.sh"
  require_idle_gpus
  CUDA_VISIBLE_DEVICES=0 HF_HOME="$work_dir/huggingface-r002" \
    PYTHONPATH="$source_dir/src" python "$source_dir/scripts/run_r002_experiment.py" \
      --stage reflex-pre \
      --artifact-dir "$artifact_dir" \
      --decider-source "$decider_dir" \
      --source-revision "$decider_revision"
  require_preexperiment_ready
}

require_preexperiment_ready() {
  python - "$artifact_dir/preexperiment-readiness.json" "$artifact_dir/calibration.json" "$artifact_dir/fan.json" <<'PY'
import json
import sys
from pathlib import Path

readiness, calibration, fan = map(Path, sys.argv[1:])
payload = json.loads(readiness.read_text(encoding="utf-8"))
if payload.get("status") != "READY" or payload.get("heldout_accessed") is not False:
    raise SystemExit("FNX-R002 pre-experiment gate failed")
if not calibration.is_file() or not fan.is_file():
    raise SystemExit("FNX-R002 pre-experiment artifacts are incomplete")
print("FNX-R002 PRE-EXPERIMENT: READY")
PY
}

publish_preparing_state() {
  python - "$FNX_PUBLIC_METRICS_PATH" <<'PY'
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "experiment_id": "FNX-R002",
    "model": "FNX-1 Reflex",
    "status": "PREPARING",
    "dry_run": False,
    "updated_at": datetime.now(UTC).isoformat(),
    "production_status": "NOT_PRODUCTION_CERTIFIED",
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

run_strong_final() {
  MODE=inference MODEL="$model" "$script_dir/run_mode.sh"
  PYTHONPATH="$source_dir/src" python "$source_dir/scripts/run_r002_experiment.py" \
    --stage strong-final \
    --artifact-dir "$artifact_dir"
}

authorize_heldout() {
  if [[ -z "$live_boundary_file" ]] || [[ ! -f "$live_boundary_file" ]]; then
    echo "FNX-R002 live boundary manifest is required before held-out access" >&2
    exit 1
  fi
  PYTHONPATH="$worker_dir" python3 -m src.r002_boundary \
    --input "$live_boundary_file" \
    --output "$artifact_dir/live-boundary.json" \
    --not-before "$not_before"
}

run_reflex_final() {
  "$script_dir/stop_ollama.sh"
  require_idle_gpus
  CUDA_VISIBLE_DEVICES=0 HF_HOME="$work_dir/huggingface-r002" \
    PYTHONPATH="$source_dir/src" python "$source_dir/scripts/run_r002_experiment.py" \
      --stage reflex-final \
      --artifact-dir "$artifact_dir" \
      --decider-source "$decider_dir" \
      --source-revision "$decider_revision"
}

persist_artifacts() {
  local archive="$work_dir/FNX-R002-private-artifacts.tar.gz"
  tar -czf "$archive" \
    -C "$artifact_dir" . \
    -C "$source_dir" reports docs/R002_PLAN.md docs/R002_PREEXPERIMENT_READINESS.md \
      docs/R002_COVERAGE_CURVE.md docs/R002_RESULTS.md \
      datasets/manifests/fnx-core-002.json \
      datasets/manifests/fnx-core-002-contamination.json
  sha256sum "$archive"
}

main() {
  cd "$worker_dir"
  require_t4_x2 >/dev/null
  prepare_source
  prepare_decider
  if [[ "$reuse_preexperiment" == "true" ]]; then
    require_preexperiment_ready
  elif [[ "$reuse_preexperiment" == "false" ]]; then
    run_reflex_pre
  else
    echo "FNX_R002_REUSE_PREEXPERIMENT must be true or false" >&2
    exit 64
  fi
  publish_preparing_state
  if [[ "$preflight_only" == "true" ]]; then
    echo "FNX-R002 PREFLIGHT ONLY: COMPLETE"
    exit 0
  elif [[ "$preflight_only" != "false" ]]; then
    echo "FNX_R002_PREFLIGHT_ONLY must be true or false" >&2
    exit 64
  fi
  authorize_heldout
  run_strong_final
  run_reflex_final
  persist_artifacts
}

main "$@"
