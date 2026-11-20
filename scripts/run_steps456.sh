#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON="${CONDA_PREFIX:-$HOME/.local/share/cs336/miniforge3/envs/CS336}/bin/python"
export WANDB_MODE=disabled
"$PYTHON" -u scripts/run_steps456.py baseline
"$PYTHON" -u scripts/run_steps456.py profiles
"$PYTHON" scripts/mixed_precision_probe.py
"$PYTHON" -u scripts/run_steps456.py memory
