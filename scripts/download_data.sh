#!/usr/bin/env bash
# Download WILDS iWildCam v2.0 (~12 GB) and the WILDS ERM checkpoint (~96 MB).
# URLs and target paths are read from common/config.yaml. Skips anything already present.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"

cfg() { "$PYTHON" -c "from common import load_config, resolve; c = load_config(); print($1)"; }

INSECURE=""  # set to --insecure once the user agrees to skip TLS verification

# download <url> <file>: fetch via a .part file; if the TLS certificate cannot be
# verified (curl exit 60; CodaLab's certificate expired on 2026-10-07), ask before
# retrying without verification.
download() {
  local rc=0 host="${1#https://}"
  host="${host%%/*}"
  curl -fL $INSECURE -o "$2.part" "$1" || rc=$?
  if [ "$rc" -eq 60 ] && [ -z "$INSECURE" ]; then
    echo >&2
    echo "WARNING: the TLS certificate of $host could not be verified" >&2
    echo "         (CodaLab's certificate has been expired since 2026-10-07)." >&2
    echo "         Without verification, the origin of the downloaded files cannot be confirmed." >&2
    local answer=""
    read -r -p "Continue without certificate verification? [y/N] " answer < /dev/tty || true
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
      echo "Aborted." >&2
      exit 1
    fi
    INSECURE="--insecure"
    rc=0
    curl -fL $INSECURE -o "$2.part" "$1" || rc=$?
  fi
  [ "$rc" -eq 0 ] || { echo "Download failed (curl exit $rc): $1" >&2; exit "$rc"; }
  mv "$2.part" "$2"
}

DATA_DIR=$(cfg 'resolve(c["paths"]["data_dir"])')
DATA_URL=$(cfg 'c["dataset"]["url"]')
CKPT=$(cfg 'resolve(c["paths"]["checkpoint"])')
CKPT_URL=$(cfg 'c["model"]["checkpoint_url"]')

if [ -f "$DATA_DIR/metadata.csv" ]; then
  echo "Dataset already in $DATA_DIR"
else
  echo "Downloading iWildCam v2.0 (~12 GB) to $DATA_DIR ..."
  mkdir -p "$DATA_DIR"
  download "$DATA_URL" "$DATA_DIR/archive.tar.gz"
  tar -xzf "$DATA_DIR/archive.tar.gz" -C "$DATA_DIR"
  rm "$DATA_DIR/archive.tar.gz"
fi

if [ -f "$CKPT" ]; then
  echo "Checkpoint already at $CKPT"
else
  echo "Downloading WILDS ERM checkpoint to $CKPT ..."
  mkdir -p "$(dirname "$CKPT")"
  download "$CKPT_URL" "$CKPT"
fi
