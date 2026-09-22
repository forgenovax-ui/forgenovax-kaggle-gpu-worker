#!/usr/bin/env bash
set -Eeuo pipefail

mode="${MODE:-inference}"
case "$mode" in
  inference)
    exec "$(dirname "$0")/start_inference_mode.sh"
    ;;
  training)
    exec "$(dirname "$0")/start_training_mode.sh"
    ;;
  *)
    echo "MODE must be exactly inference or training" >&2
    exit 64
    ;;
esac

