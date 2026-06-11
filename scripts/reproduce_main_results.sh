#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGS=(
  "${ROOT_DIR}/configs/c0_simulink_nn.yaml"
  "${ROOT_DIR}/configs/c1_lstm9d.yaml"
  "${ROOT_DIR}/configs/c2_lstm15d.yaml"
  "${ROOT_DIR}/configs/c3_bidir_attn.yaml"
)

echo "[reproduce_main_results] Starting baseline and main model experiments..."
for cfg in "${CONFIGS[@]}"; do
  echo "[reproduce_main_results] Pending execution for ${cfg}"
  echo "  - ROS launch: roslaunch drone_sim scene05_composite.launch controller_config:=${cfg}"
  echo "  - MATLAB eval: matlab -batch \"run('matlab/evaluation/compute_metrics.m')\""
done

echo "[reproduce_main_results] Fill in the concrete launch/evaluation commands for your setup."
