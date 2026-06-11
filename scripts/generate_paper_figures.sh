#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[generate_paper_figures] MATLAB batch command template:"
echo "matlab -batch \"cd('${ROOT_DIR}/matlab/evaluation'); plot_all_figures\""
