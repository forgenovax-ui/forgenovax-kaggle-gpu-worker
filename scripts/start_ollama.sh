#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${FORGENOVAX_WORK_DIR:-/kaggle/working}"
BIN_DIR="${WORK_DIR}/bin"
LOG_FILE="${WORK_DIR}/ollama.log"
PID_FILE="${WORK_DIR}/ollama.pid"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

export PATH="${BIN_DIR}:${PATH}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
# Stable workspace location supports restoring read-only Kaggle dataset snapshots.
export OLLAMA_MODELS="${OLLAMA_MODELS:-${WORK_DIR}/ollama-models}"
export OLLAMA_CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_SCHED_SPREAD="${OLLAMA_SCHED_SPREAD:-1}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-20m}"

if [[ "${OLLAMA_HOST}" != "127.0.0.1:11434" ]]; then
  echo "Refusing to start: OLLAMA_HOST must be 127.0.0.1:11434" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_indices="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd, -)"
else
  gpu_indices=""
fi
if [[ -z "$gpu_indices" ]] && [[ "${FORGENOVAX_REQUIRE_GPU:-1}" == "1" ]]; then
  echo "GPU acceleration unavailable: nvidia-smi did not report any devices" >&2
  exit 1
fi
requested_devices="${FORGENOVAX_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-$gpu_indices}}"
if [[ -n "$requested_devices" ]]; then
  if [[ ! "$requested_devices" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "FORGENOVAX_CUDA_VISIBLE_DEVICES must be comma-separated GPU indexes" >&2
    exit 64
  fi
  IFS=',' read -r -a requested_array <<< "$requested_devices"
  for requested in "${requested_array[@]}"; do
    if ! tr ',' '\n' <<< "$gpu_indices" | grep -Fxq "$requested"; then
      echo "Requested GPU $requested is not visible; detected indexes: ${gpu_indices:-none}" >&2
      exit 1
    fi
  done
  export CUDA_VISIBLE_DEVICES="$requested_devices"
  echo "Ollama CUDA visibility: $CUDA_VISIBLE_DEVICES (separate GPU memory devices)"
fi

mkdir -p "${WORK_DIR}"
if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is already ready at http://${OLLAMA_HOST}"
  exit 0
fi
rm -f "${PID_FILE}"

nohup ollama serve >"${LOG_FILE}" 2>&1 &
OLLAMA_PID=$!
echo "${OLLAMA_PID}" >"${PID_FILE}"

for _ in $(seq 1 90); do
  if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Ollama is ready at http://${OLLAMA_HOST} (PID ${OLLAMA_PID})"
    exit 0
  fi
  if ! kill -0 "${OLLAMA_PID}" 2>/dev/null; then
    echo "Ollama exited before becoming ready" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "Ollama did not become ready within 90 seconds" >&2
tail -n 80 "${LOG_FILE}" >&2 || true
exit 1
