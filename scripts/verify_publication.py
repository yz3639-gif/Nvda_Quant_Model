"""Verify the published metadata export without market-data access or dependencies."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_fingerprint(path: Path) -> tuple[int, str]:
    entries = []
    if path.suffix == ".json":
        def collect(value, key=""):
            if isinstance(value, dict):
                for name, child in value.items():
                    if name != "bytes":
                        collect(child, key + "/" + name)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    collect(child, key + "/" + str(index))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                entries.append([key, value])
        collect(json.loads(path.read_text()))
    elif path.suffix == ".csv":
        number = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf)", re.I)
        for row_index, row in enumerate(csv.reader(io.StringIO(path.read_text()))):
            for column, value in enumerate(row):
                if number.fullmatch(value):
                    entries.append([row_index, column, value])
    encoded = json.dumps(entries, sort_keys=True, allow_nan=True).encode()
    return len(entries), hashlib.sha256(encoded).hexdigest()


def main() -> None:
    record = json.loads((ROOT / "research/PUBLICATION_PROVENANCE.json").read_text())
    counts = {"publication_files": 0, "numeric_files": 0, "numeric_values": 0,
              "artifact_references": 0, "baseline_artifacts": 0,
              "replay_manifest_references": 0, "available_replay_inputs": 0,
              "unshipped_replay_inputs": 0, "original_backups_verified": 0,
              "source_file_hashes": 0}

    def require(condition, message):
        if not condition:
            raise SystemExit("FAIL: " + message)

    for item in record["files"]:
        path = ROOT / item["path"]
        require(sha256(path) == item["published_sha256"], "publication hash: " + item["path"])
        original = ROOT / "research/runs/publication_backup" / item["path"]
        if original.exists():
            require(sha256(original) == item["original_sha256"], "original backup hash: " + item["path"])
            counts["original_backups_verified"] += 1
        counts["publication_files"] += 1

    for item in record["numeric_validation"]:
        count, digest = numeric_fingerprint(ROOT / item["path"])
        require(digest == item["original_numeric_sha256"] == item["published_numeric_sha256"],
                "numeric values changed: " + item["path"])
        require(count == item["numeric_values"], "numeric count changed: " + item["path"])
        original = ROOT / "research/runs/publication_backup" / item["path"]
        if original.exists():
            require(numeric_fingerprint(original) == (count, digest), "original backup numeric values")
        counts["numeric_files"] += 1
        counts["numeric_values"] += count

    for path in sorted((ROOT / "research/results").rglob("manifest.json")):
        manifest = json.loads(path.read_text())

        def artifact_maps(value):
            if isinstance(value, dict):
                for name, child in value.items():
                    if name == "artifacts" and isinstance(child, dict):
                        for relative, expected in child.items():
                            require(sha256(path.parent / relative) == expected,
                                    "artifact hash: " + str(path.relative_to(ROOT)) + ": " + relative)
                            counts["artifact_references"] += 1
                    else:
                        artifact_maps(child)
            elif isinstance(value, list):
                for child in value:
                    artifact_maps(child)
        artifact_maps(manifest)

    baseline = json.loads((ROOT / "research/baseline/manifest.json").read_text())
    for item in baseline["historical_artifacts"]:
        path = ROOT / item["path"]
        require(sha256(path) == item["sha256"], "baseline hash: " + item["path"])
        require(path.stat().st_size == item["bytes"], "baseline size: " + item["path"])
        counts["baseline_artifacts"] += 1

    replay_dir = ROOT / "research/results/frozen_20260918/historical_replay"
    replay = json.loads((replay_dir / "historical_replay_manifest.json").read_text())
    require(sha256(replay_dir / "historical_replay.csv") == replay["summary_sha256"], "replay summary")
    for item in replay["runs"]:
        path = ROOT / item["manifest_path"]
        require(sha256(path) == item["manifest_sha256"], "replay manifest: " + item["manifest_path"])
        counts["replay_manifest_references"] += 1
        manifest = json.loads(path.read_text())
        require(manifest["source_freeze_manifest_sha256"] == sha256(ROOT / "research/baseline/manifest.json"),
                "replay source-freeze reference")
        for source in manifest["inputs"]:
            source_path = ROOT / source["path"]
            if not source_path.exists():
                require(source["path"].startswith("nvda_quant_model/cache/"), "unexpected missing replay input")
                counts["unshipped_replay_inputs"] += 1
                continue
            require(sha256(source_path) == source["sha256"], "replay input: " + source["path"])
            require(source_path.stat().st_size == source["bytes"], "replay input size")
            counts["available_replay_inputs"] += 1

    baseline_root = ROOT / "research/baseline"
    baseline_files = {str(path.relative_to(baseline_root)): sha256(path)
                      for path in sorted(baseline_root.rglob("*")) if path.is_file()}
    baseline_hash = hashlib.sha256(json.dumps(baseline_files, sort_keys=True).encode()).hexdigest()
    for relative, original in record["historical_run_identity"].items():
        manifest = json.loads((ROOT / relative).read_text())
        require(manifest["code"]["commit"] == original["source_commit"], "source commit identity")
        require(manifest["source_sha256"] == original["source_sha256"], "source hash identity")
        require(manifest["input_sha256"] == original["input_sha256"], "price input identity")
        require(manifest["cache_sha256"] == original["cache_sha256"], "price cache identity")
        if manifest["baseline_sha256"] is not None:
            require(manifest["baseline_sha256"] == baseline_hash == original["published_baseline_sha256"],
                    "published baseline aggregate")
        for source, expected in manifest["code"]["files"].items():
            require(sha256(ROOT / source) == expected, "engine source: " + source)
            counts["source_file_hashes"] += 1

    print(json.dumps({"status": "pass", **counts}, indent=2))


if __name__ == "__main__":
    main()
