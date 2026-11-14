#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-data/on006840}"
python scripts/download_iackd_manifest.py \
  --dataset-id on006840 \
  --version v1.0.0 \
  --output-dir "$DESTINATION" \
  --max-workers 4
