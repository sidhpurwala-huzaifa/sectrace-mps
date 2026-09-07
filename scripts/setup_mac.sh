#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This convenience script is for macOS. Other systems: python -m pip install -c constraints-tested-torch.txt -e '.[data,dev]'" >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Use a native arm64 Terminal/Python, not a Rosetta x86_64 environment." >&2
  exit 1
fi
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys,platform; assert sys.version_info >= (3,10); assert platform.machine() == "arm64"'
"$PYTHON" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-tested-torch.txt -e '.[data,dev]'
python -m pip freeze > installed-mac-versions.txt
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTORCH_MPS_FAST_MATH=0
python -m sectrace doctor --device mps --output mac-doctor.json
