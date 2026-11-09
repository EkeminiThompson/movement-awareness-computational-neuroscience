#!/usr/bin/env python3
"""Inventory a local BIDS dataset without loading raw EEG or motion arrays.

Usage:
  python src/audit_bids_dataset.py /path/to/ds006840 --out reports/data_audit

The script records file metadata, participant/session coverage, empty files,
and event-table row counts. It does not establish trial correspondence or
validate movement-onset semantics; those require a dataset-specific review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DATA_SUFFIXES = {".edf", ".bdf", ".vhdr", ".eeg", ".set", ".fif", ".nii", ".nii.gz", ".mat", ".tsv", ".json"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def count_tsv_rows(path: Path) -> tuple[int | None, str | None]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return sum(1 for _ in reader), None
    except (OSError, UnicodeError, csv.Error) as exc:
        return None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Root of the locally retrieved BIDS dataset")
    parser.add_argument("--out", type=Path, default=Path("reports/data_audit"))
    parser.add_argument("--hash", action="store_true", help="Calculate SHA-256 for files (can take time)")
    args = parser.parse_args()

    root = args.dataset.resolve()
    if not root.is_dir():
        parser.error(f"Dataset directory does not exist: {root}")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    participants = sorted({part for p in files for part in p.parts if part.startswith("sub-")})
    sessions = sorted({part for p in files for part in p.parts if part.startswith("ses-")})
    empty_files = [str(p.relative_to(root)) for p in files if p.stat().st_size == 0]

    records = []
    event_tables = []
    suffix_counts = Counter()
    for path in files:
        relative = str(path.relative_to(root))
        suffix = "".join(path.suffixes[-2:]) if path.name.endswith(".nii.gz") else path.suffix
        suffix_counts[suffix.lower()] += 1
        record = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "suffix": suffix.lower(),
        }
        if args.hash:
            record["sha256"] = sha256_file(path)
        if path.name.endswith("_events.tsv"):
            rows, error = count_tsv_rows(path)
            event_tables.append({"path": relative, "data_rows": rows, "error": error})
        records.append(record)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "participant_labels": participants,
        "participant_count": len(participants),
        "session_labels": sessions,
        "session_count": len(sessions),
        "empty_files": empty_files,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "event_tables": event_tables,
        "files": records,
        "limitations": [
            "File inventory does not prove that a recording contains valid signal data.",
            "Event row counts do not prove EEG-to-kinematic trial correspondence.",
            "Movement-onset semantics and pre-movement windows require task-specific validation.",
            "No raw signal arrays are loaded or analysed by this script.",
        ],
    }
    (args.out / "data_inventory.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (args.out / "file_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "suffix", "sha256"])
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})

    print(f"Files inventoried: {len(files)}")
    print(f"Participants found: {len(participants)}")
    print(f"Empty files: {len(empty_files)}")
    print(f"Event tables: {len(event_tables)}")
    print(f"Reports written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
