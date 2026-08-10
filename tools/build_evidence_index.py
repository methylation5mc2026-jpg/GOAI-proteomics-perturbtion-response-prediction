"""Build the deterministic index for the public evidence snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
INDEX = EVIDENCE / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = sorted(
        path for path in EVIDENCE.iterdir()
        if path.is_file() and path != INDEX
    )
    artifacts = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "media_type": "application/json" if path.suffix == ".json" else "text/csv",
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    payload = {
        "schema_version": 1,
        "scope": "public evidence snapshots only; runtime matrices, predictions, weights, caches and logs are excluded",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    INDEX.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"indexed {len(artifacts)} public evidence snapshots")


if __name__ == "__main__":
    main()
