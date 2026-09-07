from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import tempfile
from typing import Iterator


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_records(path: str | Path) -> Iterator[dict]:
    """Stream JSONL; stream JSON arrays when the optional ijson is available."""
    path = Path(path)
    with path.open("rb") as f:
        first = f.read(4096).lstrip()[:1]
    if first == b"[":
        try:
            import ijson
        except ImportError as e:
            raise RuntimeError("JSON arrays need `pip install -e '.[data]'`; convert to JSONL otherwise.") from e
        with path.open("rb") as f:
            for r in ijson.items(f, "item", use_float=True):
                yield r
    else:
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_number}: {e}") from e
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield record


def atomic_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count
