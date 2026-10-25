#!/usr/bin/env bash
# Restart only the authenticated gateway (uvicorn), e.g. after rotating
# FORGENOVAX_API_KEY. Ollama, the loaded model and the Cloudflare tunnel keep
# running, so the public URL does not change.
#
# In the notebook, reload the secret into the kernel environment first:
#   from kaggle_secrets import UserSecretsClient
#   os.environ["FORGENOVAX_API_KEY"] = UserSecretsClient().get_secret("FORGENOVAX_API_KEY").strip()
#   subprocess.run(["bash", "scripts/restart_proxy.sh"], check=True, env=os.environ.copy())
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${FORGENOVAX_WORK_DIR:-/kaggle/working}"
PID_FILE="${WORK_DIR}/proxy.pid"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(tr -dc '0-9' <"${PID_FILE}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    # PID reuse must never result in terminating an unrelated process.
    command_line="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    if [[ "${command_line}" != *"uvicorn src.proxy.app:app"* ]]; then
      echo "PID ${pid} is not the gateway; refusing to kill" >&2
      exit 1
    fi
    kill "${pid}"
    for _ in $(seq 1 10); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then kill -KILL "${pid}"; fi
    echo "Gateway PID ${pid} stopped"
  fi
  rm -f "${PID_FILE}"
fi

exec bash "${REPO_ROOT}/scripts/start_proxy.sh"
