#!/usr/bin/env bash
set -euo pipefail

DATASET_ID="on006840"
EXPECTED_VERSION="v1.0.0"
DESTINATION="${1:-data}"

mkdir -p "$DESTINATION"
if ! command -v nemar >/dev/null 2>&1; then
  echo "NEMAR CLI is required. Install Bun, DataLad and git-annex first." >&2
  exit 2
fi
if ! command -v datalad >/dev/null 2>&1 || ! command -v git-annex >/dev/null 2>&1; then
  echo "DataLad and git-annex are required for the complete dataset download." >&2
  exit 2
fi

cd "$DESTINATION"
if [[ ! -d "$DATASET_ID" ]]; then
  nemar dataset download "$DATASET_ID" --stimuli --derivatives
fi
cd "$DATASET_ID"

python - <<'PY'
import json
from pathlib import Path
p = Path("dataset_description.json")
if not p.exists():
    raise SystemExit("dataset_description.json missing")
d = json.loads(p.read_text(encoding="utf-8"))
print("Dataset:", d.get("Name"))
print("Version:", d.get("DatasetVersion"))
print("DOI:", d.get("DatasetDOI"))
PY

echo "Downloaded NEMAR dataset $DATASET_ID. Verify the recorded DatasetVersion is $EXPECTED_VERSION."
echo "Do not upload or commit the raw participant recordings to the analysis repository."
