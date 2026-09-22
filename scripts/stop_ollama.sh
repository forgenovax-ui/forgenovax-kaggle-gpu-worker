#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${FORGENOVAX_WORK_DIR:-/kaggle/working}"

if command -v ollama >/dev/null 2>&1; then
  ollama stop "${MODEL:-qwen3-coder:30b}" >/dev/null 2>&1 || true
fi

pid_file="${WORK_DIR}/ollama.pid"
if [[ ! -f "${pid_file}" ]]; then
  echo "ollama: no PID file"
  exit 0
fi

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
  echo "ollama: stopped PID ${pid}"
else
  echo "ollama: process already stopped"
fi
rm -f "${pid_file}"
