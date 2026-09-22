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

if [[ "${unauthorized_code}" != "401" || "${invalid_code}" != "401" || "${authorized_code}" != "200" ]]; then
  echo "Authentication preflight failed (missing=${unauthorized_code}, invalid=${invalid_code}, valid=${authorized_code})" >&2
  exit 1
fi

rm -f "${URL_FILE}"
nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PROXY_PORT}" \
  >"${LOG_FILE}" 2>&1 &
CLOUDFLARED_PID=$!
echo "${CLOUDFLARED_PID}" >"${PID_FILE}"

for _ in $(seq 1 90); do
  tunnel_url="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "${LOG_FILE}" 2>/dev/null | tail -n 1 || true)"
  if [[ -n "${tunnel_url}" ]]; then
    echo "${tunnel_url}" >"${URL_FILE}"
    echo "Cloudflare Quick Tunnel is ready: ${tunnel_url} (PID ${CLOUDFLARED_PID})"
    exit 0
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
