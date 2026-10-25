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

health_body="$(curl -fsS "http://${PROXY_HOST}:${PROXY_PORT}/healthz" 2>/dev/null || true)"
if [[ -n "$health_body" ]]; then
  if python3 -c 'import json,sys; assert json.load(sys.stdin) == {"ok":True,"service":"forgenovax-kaggle-ai"}' <<< "$health_body" &&
     curl -fsS --max-time "${READINESS_TIMEOUT_SECONDS:-180}" \
       -H "Authorization: Bearer ${FORGENOVAX_API_KEY}" \
       "http://${PROXY_HOST}:${PROXY_PORT}/readyz" >/dev/null; then
    echo "Proxy and model inference are already ready at http://${PROXY_HOST}:${PROXY_PORT}"
    exit 0
  fi
  echo "Port ${PROXY_PORT} responds but authenticated inference is not ready; run scripts/shutdown.sh before restart" >&2
  exit 1
fi
rm -f "${PID_FILE}"

nohup python3 -m uvicorn src.proxy.app:app \
  --host "${PROXY_HOST}" \
  --port "${PROXY_PORT}" \
  --no-access-log \
  >"${LOG_FILE}" 2>&1 &
PROXY_PID=$!
echo "${PROXY_PID}" >"${PID_FILE}"

for _ in $(seq 1 60); do
  if curl -fsS "http://${PROXY_HOST}:${PROXY_PORT}/healthz" >/dev/null 2>&1; then
    if curl -fsS --max-time "${READINESS_TIMEOUT_SECONDS:-180}" \
      -H "Authorization: Bearer ${FORGENOVAX_API_KEY}" \
      "http://${PROXY_HOST}:${PROXY_PORT}/readyz" >/dev/null 2>&1; then
      echo "Authenticated proxy and model inference are ready at http://${PROXY_HOST}:${PROXY_PORT} (PID ${PROXY_PID})"
      exit 0
    fi
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
