#!/usr/bin/env bash
# Run in a dedicated Python 3.11 environment. Downloads upstream code/dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
GOVSIM_COMMIT=1d11adf
if [[ -e third_party/GovSim ]]; then
  echo 'third_party/GovSim already exists; inspect it manually to preserve local changes.' >&2
  exit 1
fi
mkdir -p third_party
git clone --recurse-submodules https://github.com/giorgiopiatti/GovSim third_party/GovSim
git -C third_party/GovSim checkout "$GOVSIM_COMMIT"
git -C third_party/GovSim submodule update --init --recursive
python configs/govsim/pathfinder_local_api.patch.py third_party/GovSim
python -m pip install 'setuptools<80' torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python -m pip install -r third_party/GovSim/pathfinder/requirements.txt -r third_party/GovSim/requirements.txt regex
python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")'
git -C third_party/GovSim rev-parse HEAD
python -m pip freeze
