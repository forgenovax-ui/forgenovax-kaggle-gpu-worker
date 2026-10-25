#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${FORGENOVAX_WORK_DIR:-/kaggle/working}"
BIN_DIR="${WORK_DIR}/bin"
LOG_FILE="${WORK_DIR}/cloudflared.log"
PID_FILE="${WORK_DIR}/cloudflared.pid"
URL_FILE="${WORK_DIR}/tunnel_url"
PROXY_PORT="${PROXY_PORT:-8000}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi
export PATH="${BIN_DIR}:${PATH}"

if [[ -z "${FORGENOVAX_API_KEY:-}" ]]; then
  echo "FORGENOVAX_API_KEY is required" >&2
  exit 1
fi

listener_output=""
if command -v ss >/dev/null 2>&1; then
  listener_output="$(ss -ltnp 2>/dev/null | awk '$4 ~ /:11434$/ {print $4}')"
elif command -v lsof >/dev/null 2>&1; then
  listener_output="$(lsof -nP -iTCP:11434 -sTCP:LISTEN 2>/dev/null | awk 'NR > 1 {print $9}')"
else
  echo "Cannot verify Ollama isolation: neither ss nor lsof is available" >&2
  exit 1
fi

if [[ -z "${listener_output}" ]]; then
  echo "Cannot verify Ollama isolation: no listener found on port 11434" >&2
  exit 1
fi
if grep -Eq '(^|[[:space:]])(0\.0\.0\.0|\*|\[::\]|::):?11434' <<<"${listener_output}"; then
  echo "Unsafe Ollama listener detected: ${listener_output}" >&2
  exit 1
fi
if ! grep -Eq '127\.0\.0\.1:11434' <<<"${listener_output}"; then
  echo "Ollama is not confirmed on 127.0.0.1:11434: ${listener_output}" >&2
  exit 1
fi

unauthorized_code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PROXY_PORT}/v1/models")"
invalid_code="$(curl -sS -o /dev/null -w '%{http_code}' \
  -H 'Authorization: Bearer definitely-invalid' \
  "http://127.0.0.1:${PROXY_PORT}/v1/models")"
authorized_code="$(curl -sS -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer ${FORGENOVAX_API_KEY}" \
  "http://127.0.0.1:${PROXY_PORT}/v1/models")"
readiness_code="$(curl -sS --max-time "${READINESS_TIMEOUT_SECONDS:-180}" -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer ${FORGENOVAX_API_KEY}" \
  "http://127.0.0.1:${PROXY_PORT}/readyz")"

if [[ "${unauthorized_code}" != "401" || "${invalid_code}" != "401" || "${authorized_code}" != "200" || "${readiness_code}" != "200" ]]; then
  echo "Authentication/readiness preflight failed (missing=${unauthorized_code}, invalid=${invalid_code}, valid=${authorized_code}, ready=${readiness_code})" >&2
  exit 1
fi

# A rerun should reuse a still-working tunnel instead of leaving duplicate
# cloudflared processes running. A PID file alone is never trusted.
if [[ -f "${PID_FILE}" ]]; then
  previous_pid="$(cat "${PID_FILE}")"
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    process_command="$(ps -p "$previous_pid" -o args= 2>/dev/null || true)"
    if [[ "$process_command" == *cloudflared*tunnel* && "$process_command" == *"127.0.0.1:${PROXY_PORT}"* ]]; then
      previous_url="$(cat "${URL_FILE}" 2>/dev/null || true)"
      if [[ "$previous_url" =~ ^https://[a-z0-9-]+\.trycloudflare\.com$ ]] &&
         curl -fsS --max-time "${READINESS_TIMEOUT_SECONDS:-180}" \
           -H "Authorization: Bearer ${FORGENOVAX_API_KEY}" \
           "${previous_url}/readyz" |
           python3 -c 'import json,sys; r=json.load(sys.stdin); assert r.get("ready") is True and r.get("inference") == "verified"'; then
        echo "Existing Cloudflare tunnel is healthy: ${previous_url} (PID ${previous_pid})"
        exit 0
      fi
      echo "Replacing stale Cloudflare tunnel PID ${previous_pid}"
      kill "$previous_pid" 2>/dev/null || true
      for _ in {1..10}; do
        kill -0 "$previous_pid" 2>/dev/null || break
        sleep 1
      done
    else
      echo "Ignoring stale Cloudflare PID file: PID belongs to a different process" >&2
    fi
  fi
fi

rm -f "${PID_FILE}" "${URL_FILE}"
nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PROXY_PORT}" \
  >"${LOG_FILE}" 2>&1 &
CLOUDFLARED_PID=$!
echo "${CLOUDFLARED_PID}" >"${PID_FILE}"

for _ in $(seq 1 90); do
  tunnel_url="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "${LOG_FILE}" 2>/dev/null | tail -n 1 || true)"
  if [[ -n "${tunnel_url}" ]]; then
    echo "${tunnel_url}" >"${URL_FILE}"
    if curl -fsS --max-time "${READINESS_TIMEOUT_SECONDS:-180}" \
      -H "Authorization: Bearer ${FORGENOVAX_API_KEY}" \
      "${tunnel_url}/readyz" >/dev/null 2>&1; then
      echo "Cloudflare Quick Tunnel and public inference are ready: ${tunnel_url} (PID ${CLOUDFLARED_PID})"
      exit 0
    fi
  fi
  if ! kill -0 "${CLOUDFLARED_PID}" 2>/dev/null; then
    echo "cloudflared exited before publishing a tunnel URL" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "No Cloudflare Quick Tunnel URL appeared within 90 seconds" >&2
tail -n 80 "${LOG_FILE}" >&2 || true
exit 1
