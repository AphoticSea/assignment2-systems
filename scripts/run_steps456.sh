#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON="${CONDA_PREFIX:-$HOME/.local/share/cs336/miniforge3/envs/CS336}/bin/python"
export WANDB_MODE=disabled
mkdir -p "results/steps456/${CS336_RUN_NAME:-v3}"
"$PYTHON" scripts/check_environment.py --output "results/steps456/${CS336_RUN_NAME:-v3}/environment.json"
"$PYTHON" scripts/capacity_probe.py
"$PYTHON" -u scripts/run_steps456.py baseline
"$PYTHON" -u scripts/run_steps456.py profiles
"$PYTHON" scripts/mixed_precision_probe.py
"$PYTHON" -u scripts/run_steps456.py memory
