#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[run_all_experiments] Running main results pipeline..."
bash "${ROOT_DIR}/scripts/reproduce_main_results.sh"

echo "[run_all_experiments] Running ablation pipeline..."
bash "${ROOT_DIR}/scripts/reproduce_ablation.sh"

echo "[run_all_experiments] Generating paper figures..."
bash "${ROOT_DIR}/scripts/generate_paper_figures.sh"

echo "[run_all_experiments] Completed."
