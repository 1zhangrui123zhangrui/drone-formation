#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$ROOT_DIR/data/processed}"
MODEL_DIR="${2:-$ROOT_DIR/data/trained_models}"
LOG_DIR="${3:-$ROOT_DIR/results/training}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/matlab_train_$(date +%Y%m%d_%H%M%S).log"

if ! command -v "$MATLAB_BIN" >/dev/null 2>&1; then
  echo "MATLAB executable not found: $MATLAB_BIN" >&2
  echo "Set MATLAB_BIN=/path/to/matlab or run this script on a MATLAB-enabled machine." >&2
  exit 1
fi

MATLAB_ROOT_ESCAPED="${ROOT_DIR//\'/\'\'}"
DATA_DIR_ESCAPED="${DATA_DIR//\'/\'\'}"
MODEL_DIR_ESCAPED="${MODEL_DIR//\'/\'\'}"

"$MATLAB_BIN" -batch "cd('${MATLAB_ROOT_ESCAPED}'); addpath(fullfile(pwd,'matlab','train')); train_all_models('${DATA_DIR_ESCAPED}','${MODEL_DIR_ESCAPED}');" | tee "$LOG_FILE"

echo "MATLAB training log saved to $LOG_FILE"
