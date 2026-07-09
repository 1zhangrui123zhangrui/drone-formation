#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv-train}"
REQ_FILE="$ROOT_DIR/requirements-train-py38.txt"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$REQ_FILE"

python - <<'PY'
import numpy
import scipy
import torch
import yaml
from tqdm import tqdm

print("Training environment ready.")
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("yaml", yaml.__version__)
print("tqdm", tqdm.__version__)
PY
