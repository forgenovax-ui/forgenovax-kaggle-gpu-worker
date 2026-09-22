#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${FORGENOVAX_WORK_DIR:-/kaggle/working}"
BIN_DIR="${WORK_DIR}/bin"

mkdir -p "${BIN_DIR}"

python3 -m pip install --disable-pip-version-check -r "${REPO_ROOT}/requirements.txt"

if ! command -v ollama >/dev/null 2>&1; then
  if ! command -v zstd >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd
  fi
  curl -fsSL https://ollama.com/install.sh | sh
fi
ollama --version

if ! command -v cloudflared >/dev/null 2>&1 && [[ ! -x "${BIN_DIR}/cloudflared" ]]; then
  case "$(uname -m)" in
    x86_64|amd64) CLOUDFLARED_ARCH="amd64" ;;
    aarch64|arm64) CLOUDFLARED_ARCH="arm64" ;;
    *) echo "Unsupported architecture for cloudflared: $(uname -m)" >&2; exit 1 ;;
  esac
  curl -fL --retry 3 \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CLOUDFLARED_ARCH}" \
    -o "${BIN_DIR}/cloudflared"
  chmod 0755 "${BIN_DIR}/cloudflared"
fi

export PATH="${BIN_DIR}:${PATH}"
cloudflared --version
