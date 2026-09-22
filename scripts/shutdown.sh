#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${FORGENOVAX_WORK_DIR:-/kaggle/working}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

stop_from_pid_file() {
  local name="$1"
  local pid_file="$2"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no PID file"
    return
  fi
  local pid
  pid="$(tr -dc '0-9' <"${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    for _ in $(seq 1 10); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}"
    fi
    echo "${name}: stopped PID ${pid}"
  else
    echo "${name}: process already stopped"
  fi
  rm -f "${pid_file}"
}

stop_from_pid_file "cloudflared" "${WORK_DIR}/cloudflared.pid"
stop_from_pid_file "uvicorn" "${WORK_DIR}/proxy.pid"

if command -v ollama >/dev/null 2>&1; then
  ollama stop "${MODEL:-qwen3-coder:30b}" >/dev/null 2>&1 || true
fi
stop_from_pid_file "ollama" "${WORK_DIR}/ollama.pid"
rm -f "${WORK_DIR}/tunnel_url"

cat <<'EOF'

IMPORTANT:
Also stop the Kaggle notebook session from the Kaggle interface.
Otherwise GPU quota may continue being consumed.
EOF
