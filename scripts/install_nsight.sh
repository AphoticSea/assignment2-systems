#!/usr/bin/env bash
set -euo pipefail
DEST="$HOME/.local/share/cs336/nsight"
mkdir -p "$DEST"
cd "$DEST"
curl -fL --retry 2 -o nsight-cli.deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/NsightSystems-linux-cli-public-2026.6.1.251-39333969.deb
echo '3455b6832266542fcf3b5a4c60a3abcdd1340444c7307c9ad20731efbeb0686a  nsight-cli.deb' | sha256sum -c -
dpkg-deb -x nsight-cli.deb extracted
"$DEST/extracted/opt/nvidia/nsight-systems-cli/2026.6.1/target-linux-x64/nsys" --version
