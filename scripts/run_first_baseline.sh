#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(dirname -- "$ROOT")
PYTHON="${CONDA_PREFIX:-$HOME/.local/share/cs336/miniforge3/envs/CS336}/bin/python"
cd "$ROOT"
mkdir -p results
export WANDB_MODE=disabled
"$PYTHON" scripts/check_environment.py --output results/environment.json
for MODE in forward forward-backward train; do
    "$PYTHON" -m cs336_systems.benchmark \
        --size small --batch-size 4 --context-length 128 --dtype fp32 \
        --mode "$MODE" --warmup 5 --iterations 10 \
        --output "results/small_ctx128_fp32_${MODE}.json"
done
