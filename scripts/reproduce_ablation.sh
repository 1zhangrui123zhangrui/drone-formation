#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATIONS=(
  "${ROOT_DIR}/configs/c1_lstm9d.yaml"
  "${ROOT_DIR}/configs/c2_lstm15d.yaml"
  "${ROOT_DIR}/configs/c3_bidir_attn.yaml"
)

echo "[reproduce_ablation] Starting ablation study..."
for cfg in "${ABLATIONS[@]}"; do
  echo "[reproduce_ablation] Pending ablation for ${cfg}"
done

echo "[reproduce_ablation] Add the exact ablation loops, seeds and scenario matrix here."
