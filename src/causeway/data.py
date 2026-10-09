"""Acquisition and loading of the immutable raw dataset.

The raw file is downloaded once, accepted only if its SHA-256 matches a configured source, and then
made read-only. It is never edited: every later stage reads it and writes elsewhere.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import stat
import tempfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from causeway.config import DataConfig, DataPaths, Source

PROVENANCE_SUFFIX = ".source.json"


class ChecksumError(RuntimeError):
    """A file's content does not match the checksum recorded for it."""


class AcquisitionError(RuntimeError):
    """No configured source produced a verified raw file."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _download(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - URLs come from configs/data.toml
        return response.read()


def _fetch_verified(source: Source) -> bytes:
    payload = _download(source.url)
    if source.compression == "gzip":
        if source.sha256_compressed and sha256_bytes(payload) != source.sha256_compressed:
            raise ChecksumError(f"{source.name}: compressed checksum mismatch")
        payload = gzip.decompress(payload)
    elif source.compression != "none":
        raise ValueError(f"{source.name}: unsupported compression {source.compression!r}")
    digest = sha256_bytes(payload)
    if digest != source.sha256:
        raise ChecksumError(f"{source.name}: expected sha256 {source.sha256}, got {digest}")
    return payload


def _make_read_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def raw_path(config: DataConfig, paths: DataPaths) -> Path:
    return paths.raw / config.raw_filename


def acquire_raw(config: DataConfig, paths: DataPaths) -> tuple[Path, dict]:
    """Return the verified raw file and its provenance, downloading only if it is absent.

    An existing raw file is never overwritten. If its checksum no longer matches the source it came
    from, a ChecksumError is raised so that the modification is investigated rather than hidden.
    """
    target = raw_path(config, paths)
    provenance_file = target.with_name(target.name + PROVENANCE_SUFFIX)

    if target.exists():
        provenance = json.loads(provenance_file.read_text()) if provenance_file.exists() else {}
        known = {s.name: s.sha256 for s in config.sources}
        digest = sha256_file(target)
        expected = known.get(provenance.get("source", ""))
        if expected is None:
            matches = [name for name, sha in known.items() if sha == digest]
            if not matches:
                raise ChecksumError(f"{target} matches no configured source; remove it and re-run")
            provenance = {"source": matches[0], "sha256": digest}
        elif digest != expected:
            raise ChecksumError(
                f"{target} was modified after download (sha256 {digest}, expected {expected}). "
                "Raw data are immutable: delete the file and re-run `make data`."
            )
        return target, provenance

    target.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for source in config.sources:
        try:
            payload = _fetch_verified(source)
        except (OSError, ChecksumError, ValueError) as error:
            failures.append(f"{source.name}: {error}")
            continue
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.replace(temporary, target)
        _make_read_only(target)
        provenance = {
            "source": source.name,
            "url": source.url,
            "sha256": source.sha256,
            "bytes": len(payload),
            "failed_sources": failures,
            "source_record": asdict(source),
        }
        provenance_file.write_text(json.dumps(provenance, indent=2) + "\n")
        return target, provenance

    raise AcquisitionError("no source produced a verified raw file:\n  " + "\n  ".join(failures))


def read_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV exactly as text.

    Every field stays a string and no value is converted to NaN, so validation sees the file as it is
    (an empty field stays '' rather than silently becoming missing).
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
