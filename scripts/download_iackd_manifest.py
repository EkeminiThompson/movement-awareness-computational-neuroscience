#!/usr/bin/env python3
"""Download a pinned NEMAR release from its public manifest and verify checksums."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path, PurePosixPath


def digest_file(path: Path, algorithm: str) -> str:
    if algorithm == "git":
        h = hashlib.sha1()
        h.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    elif algorithm in {"sha256", "md5"}:
        h = hashlib.new(algorithm)
    else:
        raise ValueError(f"Unsupported NEMAR checksum algorithm: {algorithm}")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    destination = root.joinpath(*p.parts)
    if root.resolve() not in destination.resolve().parents:
        raise ValueError(f"Manifest path escapes output directory: {relative}")
    return destination


def download_one(entry: dict, root: Path) -> dict:
    rel = entry["path"]
    destination = safe_path(root, rel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(entry.get("size", 0))
    algorithm = str(entry.get("checksum_algorithm", "")).lower()
    checksum = str(entry.get("checksum", "")).lower()
    url = entry.get("url") or entry.get("bytes_url")
    if not url:
        raise ValueError(f"No stable URL in manifest entry: {rel}")

    if destination.exists() and destination.stat().st_size == expected_size:
        if not checksum or digest_file(destination, algorithm) == checksum:
            return {"path": rel, "status": "verified-existing", "size": expected_size}
        destination.unlink()

    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "5", "--retry-delay", "2", "--retry-all-errors",
        "--continue-at", "-", "--output", str(destination), url,
    ]
    subprocess.run(command, check=True)
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{rel}: expected {expected_size} bytes, got {actual_size}")
    actual_checksum = digest_file(destination, algorithm) if checksum else ""
    if checksum and actual_checksum != checksum:
        destination.unlink(missing_ok=True)
        raise ValueError(f"{rel}: {algorithm} checksum mismatch")
    return {"path": rel, "status": "downloaded-and-verified", "size": actual_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="on006840")
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--output-dir", type=Path, default=Path("data/on006840"))
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    if args.max_workers < 1 or args.max_workers > 8:
        raise ValueError("--max-workers must be between 1 and 8")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_url = (
        f"https://data.nemar.org/{args.dataset_id}/{args.version}/manifest.json"
    )
    with urllib.request.urlopen(manifest_url, timeout=60) as response:
        manifest_bytes = response.read()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("NEMAR manifest must be a non-empty JSON array")
    (root / "nemar_manifest.json").write_bytes(manifest_bytes)

    total_bytes = sum(int(item.get("size", 0)) for item in manifest)
    free_bytes = shutil.disk_usage(root).free
    if total_bytes > free_bytes * 0.90:
        raise RuntimeError(
            f"Manifest requires {total_bytes:,} bytes but only {free_bytes:,} bytes are free. "
            "Free disk space or use a larger runner."
        )
    print(
        f"Downloading {len(manifest)} files ({total_bytes / 1e9:.2f} GB) "
        f"from {args.dataset_id}@{args.version} with {args.max_workers} workers."
    )

    results = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(download_one, entry, root): entry for entry in manifest}
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"path": entry.get("path"), "error": str(exc)})
                print(f"FAILED {entry.get('path')}: {exc}")

    summary = {
        "dataset_id": args.dataset_id,
        "version": args.version,
        "manifest_url": manifest_url,
        "manifest_files": len(manifest),
        "total_bytes": total_bytes,
        "completed_files": len(results),
        "failed_files": failures,
        "results": sorted(results, key=lambda x: x["path"]),
    }
    (root / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} files failed; see download_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
