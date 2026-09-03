from __future__ import annotations

import hashlib
import os


def file_sha256(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    resolved = os.path.abspath(os.path.expanduser(path))
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
