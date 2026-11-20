#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(dirname -- "$ROOT")
CONDA_BIN=${CONDA_EXE:-"$HOME/.local/share/cs336/miniforge3/bin/conda"}
if [[ ! -x "$CONDA_BIN" ]]; then
    echo "Linux Conda was not found: $CONDA_BIN" >&2
    exit 1
fi
ENV_PREFIX="$("$CONDA_BIN" info --base)/envs/CS336"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    "$CONDA_BIN" env create --yes --prefix "$ENV_PREFIX" --file "$ROOT/environments/conda/environment.yml"
fi
# Conda owns the Python environment; uv is used only as a pip-compatible installer.
cd "$ROOT"
uv pip install --python "$ENV_PREFIX/bin/python" --torch-backend=cu128 --editable . --editable ./cs336-basics
"$ENV_PREFIX/bin/python" -m pip check
mkdir -p "$ROOT/results"
"$ENV_PREFIX/bin/python" -m pip freeze > "$ROOT/results/python-packages.txt"
"$CONDA_BIN" list --prefix "$ENV_PREFIX" --explicit > "$ROOT/results/conda-explicit.txt"
echo "Ready: $ENV_PREFIX/bin/python"
