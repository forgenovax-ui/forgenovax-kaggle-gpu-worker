#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work_dir="${FORGENOVAX_WORK_DIR:-/kaggle/working}"
artifact_dir="${FNX_R002_ARTIFACT_DIR:-$work_dir/fnx-r002-artifacts}"
metrics_path="${FNX_PUBLIC_METRICS_PATH:-$artifact_dir/live/metrics.json}"
kill_switch_path="${FNX_PUBLIC_METRICS_KILL_SWITCH_PATH:-$artifact_dir/live/PUBLIC_DISABLED}"
frozen_runner="${FNX_R002_FROZEN_RUNNER:-$script_dir/run_r002_frozen.sh}"
mode_runner="${FNX_R002_MODE_RUNNER:-$script_dir/run_mode.sh}"
shutdown_runner="${FNX_R002_SHUTDOWN_RUNNER:-$script_dir/shutdown.sh}"
curl_bin="${FNX_R002_CURL_BIN:-curl}"
probe="$(mktemp "$work_dir/fnx-r002-public-metrics.XXXXXX.json")"
trap 'rm -f "$probe"' EXIT

if [[ -e "$kill_switch_path" ]]; then
  echo "FNX-R002 public telemetry emergency kill switch is active" >&2
  exit 1
fi

export FNX_R002_ARTIFACT_DIR="$artifact_dir"
export FNX_PUBLIC_METRICS_PATH="$metrics_path"
export FNX_PUBLIC_METRICS_KILL_SWITCH_PATH="$kill_switch_path"
export FNX_R002_PREFLIGHT_ONLY=true

"$frozen_runner"

# A retained proxy may have inherited an older metrics path. Restart every live
# service so the public route is bound to this attempt's authoritative artifact.
"$shutdown_runner" >/dev/null
MODE=inference "$mode_runner"

tunnel_file="$work_dir/tunnel_url"
if [[ ! -f "$tunnel_file" ]]; then
  echo "FNX-R002 live preflight did not produce a tunnel URL" >&2
  exit 1
fi
tunnel_url="$(head -n 1 "$tunnel_file")"
if [[ ! "$tunnel_url" =~ ^https://[-a-z0-9]+\.trycloudflare\.com$ ]]; then
  echo "FNX-R002 live preflight produced an invalid tunnel URL" >&2
  exit 1
fi

metrics_url="$tunnel_url/fnx/live/metrics"
"$curl_bin" --fail --silent --show-error --max-time 30 \
  --output "$probe" "$metrics_url"
python3 - "$probe" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "experiment_id": "FNX-R002",
    "status": "PREPARING",
    "dry_run": False,
    "production_status": "NOT_PRODUCTION_CERTIFIED",
}
failed = [key for key, value in expected.items() if payload.get(key) != value]
if failed:
    raise SystemExit("FNX-R002 public metrics preflight failed: " + ", ".join(failed))
print("FNX-R002 PUBLIC TELEMETRY: READY")
PY

echo "FNX_R002_PUBLIC_TELEMETRY_URL=$metrics_url"
echo "FNX-R002 LIVE PREFLIGHT: COMPLETE — HELD-OUT CLOSED"
