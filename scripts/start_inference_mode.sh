#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_dir"

"$script_dir/install.sh"
"$script_dir/start_ollama.sh"
model="${MODEL:-qwen3-coder:30b}"
ollama pull "$model"

# Pulling verifies the model artifact, but Ollama does not report it as loaded
# until the first generation request completes. Warm it through the loopback
# API before the readiness gate so the health check measures the runtime that
# will actually serve the experiment.
curl --fail --silent --show-error --max-time 900 \
  "http://127.0.0.1:11434/api/generate" \
  -H 'Content-Type: application/json' \
  --data-binary "$(MODEL_TO_WARM="$model" python3 - <<'PY'
import json
import os

print(json.dumps({
    "model": os.environ["MODEL_TO_WARM"],
    "prompt": "Reply only with READY.",
    "stream": False,
    "keep_alive": "20m",
    "options": {"num_predict": 1},
}))
PY
)" >/dev/null

"$script_dir/start_proxy.sh"
"$script_dir/start_tunnel.sh"

# A Quick Tunnel can publish its URL before public DNS and the edge route are
# ready. Retry the complete evidence-based status gate for a bounded period;
# do not weaken or skip any individual check.
status_report="${FORGENOVAX_WORK_DIR:-/kaggle/working}/startup-status.txt"
for _ in $(seq 1 30); do
  if "$script_dir/status.sh" >"$status_report"; then
    cat "$status_report"
    exit 0
  fi
  sleep 2
done

cat "$status_report" >&2
exit 1
