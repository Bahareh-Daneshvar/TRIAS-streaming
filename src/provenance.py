from __future__ import annotations
from pathlib import Path
import hashlib


def git_blob_sha1(path: Path) -> str:
    """Return Git blob SHA-1 for the exact file bytes.

    This lets us verify that a locally supplied CSV is byte-identical to the
    public GitHub blob used in the remote audit, without trusting filenames.
    """
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
