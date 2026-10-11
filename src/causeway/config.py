"""Project paths and configuration loading."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class DataPaths:
    """Locations of every pipeline stage. Tests point `root` at a temporary directory."""

    root: Path = PROJECT_ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports" / "validation"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str
    compression: str = "none"
    sha256_compressed: str | None = None


@dataclass(frozen=True)
class DataConfig:
    name: str
    citation: str
    documentation_url: str
    expected_rows: int | None
    raw_filename: str
    sources: list[Source] = field(default_factory=list)


def load_data_config(path: Path = CONFIG_DIR / "data.toml") -> DataConfig:
    spec = tomllib.loads(path.read_text())
    dataset = spec["dataset"]
    return DataConfig(
        name=dataset["name"],
        citation=dataset["citation"],
        documentation_url=dataset["documentation_url"],
        expected_rows=dataset.get("expected_rows"),
        raw_filename=dataset["raw_filename"],
        sources=[Source(**source) for source in spec["sources"]],
    )


# Checksums recorded when each benchmark specification was frozen (line endings normalised to LF).
FROZEN_BENCHMARK_SHA256 = {1: "d88ce1a03de54eed5c8a38e095a49d01c0e07f0d12f95e66d3554f23b6f6697e"}


class FrozenSpecificationError(RuntimeError):
    """A frozen benchmark specification no longer matches the checksum recorded at freeze time."""


def load_benchmark_config(version: int = 1) -> dict:
    """Load a frozen benchmark specification, refusing one that changed after it was frozen."""
    raw = (CONFIG_DIR / "benchmark" / f"v{version}.toml").read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FROZEN_BENCHMARK_SHA256.get(version):
        raise FrozenSpecificationError(
            f"benchmark v{version} has sha256 {digest}, not the frozen value; create a new version instead"
        )
    return tomllib.loads(raw.decode())
