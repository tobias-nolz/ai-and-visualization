#!/usr/bin/env bash
# Download WILDS iWildCam v2.0 (~12 GB) and the WILDS ERM checkpoint (~96 MB).
# URLs and target paths are read from common/config.yaml. Skips anything already present.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"

cfg() { "$PYTHON" -c "from common import load_config, resolve; c = load_config(); print($1)"; }

DATA_DIR=$(cfg 'resolve(c["paths"]["data_dir"])')
DATA_URL=$(cfg 'c["dataset"]["url"]')
CKPT=$(cfg 'resolve(c["paths"]["checkpoint"])')
CKPT_URL=$(cfg 'c["model"]["checkpoint_url"]')

if [ -f "$DATA_DIR/metadata.csv" ]; then
  echo "Dataset already in $DATA_DIR"
else
  echo "Downloading iWildCam v2.0 (~12 GB) to $DATA_DIR ..."
  mkdir -p "$DATA_DIR"
  curl -fL -o "$DATA_DIR/archive.tar.gz" "$DATA_URL"
  tar -xzf "$DATA_DIR/archive.tar.gz" -C "$DATA_DIR"
  rm "$DATA_DIR/archive.tar.gz"
fi

if [ -f "$CKPT" ]; then
  echo "Checkpoint already at $CKPT"
else
  echo "Downloading WILDS ERM checkpoint to $CKPT ..."
  mkdir -p "$(dirname "$CKPT")"
  curl -fL -o "$CKPT" "$CKPT_URL"
fi
