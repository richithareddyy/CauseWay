"""Provenance stamps for analysis outputs, so results from different runs are never combined by mistake."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from causeway.config import CONFIG_DIR, PROJECT_ROOT, DataPaths
from causeway.pipeline import environment

PROVENANCE_FILE = "provenance.json"


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--", "src", "configs")
    return {
        "commit": run("rev-parse", "HEAD"),
        "uncommitted_changes": None if status is None else bool(status),
    }


def build_provenance(paths: DataPaths, spec: dict | None = None, **extra) -> dict:
    manifest = json.loads((paths.processed / "manifest.json").read_text())
    record = {
        "data": {
            "raw_sha256": manifest["raw"]["sha256"],
            "processed_sha256": manifest["processed"]["sha256"],
            "rows": manifest["processed"]["rows"],
        },
        "code": _git_state(),
        "environment": environment(),
        **extra,
    }
    if spec is not None:
        version = int(spec["meta"]["benchmark_version"].split(".")[0])
        record["benchmark"] = {
            "version": spec["meta"]["benchmark_version"],
            "sha256": _sha256_lf(CONFIG_DIR / "benchmark" / f"v{version}.toml"),
            "base_seed": spec["randomness"]["base_seed"],
            "n_draws": spec["randomness"]["n_draws"],
        }
    return record


def write_provenance(results_dir: Path, paths: DataPaths, spec: dict | None = None, **extra) -> dict:
    record = build_provenance(paths, spec, **extra)
    (results_dir / PROVENANCE_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return record


def read_provenance(results_dir: Path) -> dict:
    path = results_dir / PROVENANCE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; regenerate the outputs with `make analysis`")
    return json.loads(path.read_text())
