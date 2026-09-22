#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${FORGENOVAX_WORK_DIR:-/kaggle/working}"
LOG_FILE="${WORK_DIR}/proxy.log"
PID_FILE="${WORK_DIR}/proxy.pid"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8000}"
if [[ "${PROXY_HOST}" != "127.0.0.1" ]]; then
  echo "Refusing to start: PROXY_HOST must be 127.0.0.1" >&2
  exit 1
fi
if [[ -z "${FORGENOVAX_API_KEY:-}" || "${#FORGENOVAX_API_KEY}" -lt 24 ]]; then
  echo "FORGENOVAX_API_KEY must be set to at least 24 characters" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"
cd "${REPO_ROOT}"

if curl -fsS "http://${PROXY_HOST}:${PROXY_PORT}/healthz" >/dev/null 2>&1; then
  echo "Proxy is already ready at http://${PROXY_HOST}:${PROXY_PORT}"
  exit 0
fi

nohup python3 -m uvicorn src.proxy.app:app \
  --host "${PROXY_HOST}" \
  --port "${PROXY_PORT}" \
  --no-access-log \
  >"${LOG_FILE}" 2>&1 &
PROXY_PID=$!
echo "${PROXY_PID}" >"${PID_FILE}"

for _ in $(seq 1 60); do
  if curl -fsS "http://${PROXY_HOST}:${PROXY_PORT}/healthz" >/dev/null 2>&1; then
    echo "Authenticated proxy is ready at http://${PROXY_HOST}:${PROXY_PORT} (PID ${PROXY_PID})"
    exit 0
  fi
  if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
    echo "Proxy exited before becoming ready" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "Proxy did not become ready within 60 seconds" >&2
tail -n 80 "${LOG_FILE}" >&2 || true
exit 1
