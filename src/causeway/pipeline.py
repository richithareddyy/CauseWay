"""The `make data` pipeline: acquire -> validate raw -> interim -> processed -> validate processed."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import pandas as pd

from causeway.config import DataConfig, DataPaths
from causeway.data import acquire_raw, read_raw, sha256_file
from causeway.processing import TransformationLog, build_interim, build_processed
from causeway.validation import ValidationError, ValidationReport, validate_processed, validate_raw

INTERIM_FILE = "hillstrom_typed.parquet"
PROCESSED_FILE = "hillstrom.parquet"
TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "dowhy",
    "econml",
    "doubleml",
]


@dataclass
class PipelineResult:
    raw_path: Path
    processed_path: Path
    raw_report: ValidationReport
    processed_report: ValidationReport
    transformations: TransformationLog
    manifest: dict


def environment() -> dict:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "packages": versions}


def _write_reports(paths: DataPaths, reports: list[ValidationReport], log: TransformationLog) -> None:
    paths.reports.mkdir(parents=True, exist_ok=True)
    for report in reports:
        (paths.reports / f"validation_{report.stage}.json").write_text(report.to_json())
    (paths.reports / "transformations.json").write_text(json.dumps(log.to_records(), indent=2) + "\n")
    sections = ["# Data validation report", ""]
    sections += [report.to_markdown() for report in reports]
    if log.entries:
        sections += ["## Transformations", "", log.to_markdown()]
    (paths.reports / "validation_report.md").write_text("\n".join(sections))


def run_data_pipeline(config: DataConfig, paths: DataPaths) -> PipelineResult:
    raw_file, provenance = acquire_raw(config, paths)
    raw = read_raw(raw_file)

    raw_report = validate_raw(raw, expected_rows=config.expected_rows)
    log = TransformationLog()
    if not raw_report.passed:
        _write_reports(paths, [raw_report], log)
        raise ValidationError(raw_report)

    paths.interim.mkdir(parents=True, exist_ok=True)
    paths.processed.mkdir(parents=True, exist_ok=True)
    interim = build_interim(raw, log)
    interim.to_parquet(paths.interim / INTERIM_FILE, index=False)

    processed = build_processed(interim, log)
    processed_report = validate_processed(processed, n_raw_rows=len(raw))
    _write_reports(paths, [raw_report, processed_report], log)
    if not processed_report.passed:
        raise ValidationError(processed_report)

    processed_path = paths.processed / PROCESSED_FILE
    processed.to_parquet(processed_path, index=False)

    manifest = {
        "dataset": config.name,
        "citation": config.citation,
        "raw": {
            "path": str(raw_file.relative_to(paths.root)),
            "source": provenance.get("source"),
            "sha256": sha256_file(raw_file),
            "rows": len(raw),
            "columns": raw.shape[1],
        },
        "processed": {
            "path": str(processed_path.relative_to(paths.root)),
            "sha256": sha256_file(processed_path),
            "rows": len(processed),
            "columns": processed.shape[1],
            "column_names": list(processed.columns),
        },
        "rows_removed": len(raw) - len(processed),
        "n_transformations": len(log.entries),
        "environment": environment(),
    }
    (paths.processed / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return PipelineResult(raw_file, processed_path, raw_report, processed_report, log, manifest)


def load_processed(paths: DataPaths | None = None) -> pd.DataFrame:
    """Load the analysis dataset produced by `make data`."""
    paths = paths or DataPaths()
    path = paths.processed / PROCESSED_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make data` first")
    return pd.read_parquet(path)
