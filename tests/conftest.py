"""Shared fixtures: a small synthetic file with the Hillstrom schema.

Synthetic data are used only to test the pipeline's mechanics; analyses use the real dataset.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causeway import schema
from causeway.config import DataConfig, DataPaths, Source


def make_raw_frame(n_per_arm: int = 30, seed: int = 7) -> pd.DataFrame:
    """Valid raw-format data (all strings), including the 'Surburban' spelling and one duplicate pair."""
    rng = np.random.default_rng(seed)
    n = 3 * n_per_arm
    segments = list(schema.HISTORY_SEGMENTS)
    rows = []
    for arm in np.repeat(schema.ARMS, n_per_arm)[rng.permutation(n)]:
        bin_label = segments[rng.integers(len(segments))]
        low, high = schema.HISTORY_SEGMENTS[bin_label]
        history = round(rng.uniform(max(low, 29.99) + 0.01, min(high, 3000.0) - 0.01), 2)
        mens, womens = [(1, 0), (0, 1), (1, 1)][rng.integers(3)]
        visit = int(rng.random() < 0.3)
        conversion = int(visit and rng.random() < 0.4)
        spend = f"{rng.uniform(29.99, 499.0):.2f}" if conversion else "0"
        rows.append(
            {
                "recency": str(rng.integers(1, 13)),
                "history_segment": bin_label,
                "history": f"{history:.2f}",
                "mens": str(mens),
                "womens": str(womens),
                "zip_code": ["Urban", "Surburban", "Rural"][rng.integers(3)],
                "newbie": str(rng.integers(2)),
                "channel": schema.CHANNELS[rng.integers(3)],
                "segment": arm,
                "visit": str(visit),
                "conversion": str(conversion),
                "spend": spend,
            }
        )
    frame = pd.DataFrame(rows, columns=schema.COLUMNS)
    frame.iloc[-1] = frame.iloc[0]  # an exact duplicate row, which must be kept
    return frame


def write_csv(frame: pd.DataFrame, path: Path) -> str:
    """Write CRLF-terminated CSV like the original source and return its SHA-256."""
    path.write_bytes(frame.to_csv(index=False, lineterminator="\r\n").encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def source_file(tmp_path: Path, raw_frame: pd.DataFrame) -> tuple[Path, str]:
    source_dir = tmp_path / "remote"
    source_dir.mkdir()
    path = source_dir / "hillstrom.csv"
    return path, write_csv(raw_frame, path)


@pytest.fixture
def gzip_source(tmp_path: Path, source_file: tuple[Path, str]) -> Source:
    path, digest = source_file
    packed = path.with_suffix(".csv.gz")
    packed.write_bytes(gzip.compress(path.read_bytes()))
    return Source(
        name="local_mirror",
        url=packed.as_uri(),
        sha256=digest,
        compression="gzip",
        sha256_compressed=hashlib.sha256(packed.read_bytes()).hexdigest(),
    )


@pytest.fixture
def data_config(source_file: tuple[Path, str]) -> DataConfig:
    path, digest = source_file
    return DataConfig(
        name="synthetic test data",
        citation="test fixture",
        documentation_url="",
        expected_rows=None,
        raw_filename="hillstrom_test.csv",
        sources=[Source(name="local", url=path.as_uri(), sha256=digest)],
    )


@pytest.fixture
def paths(tmp_path: Path) -> DataPaths:
    return DataPaths(root=tmp_path / "project")
