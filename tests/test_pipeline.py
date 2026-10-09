"""End-to-end `make data` pipeline on the synthetic fixture."""

import dataclasses
import json

import pandas as pd
import pytest

from causeway.data import sha256_file
from causeway.pipeline import PROCESSED_FILE, load_processed, run_data_pipeline
from causeway.validation import ValidationError
from conftest import write_csv


def test_pipeline_produces_processed_data_reports_and_manifest(data_config, paths, raw_frame):
    result = run_data_pipeline(data_config, paths)
    raw_digest = sha256_file(result.raw_path)

    processed = load_processed(paths)
    assert len(processed) == len(raw_frame)
    assert result.manifest["rows_removed"] == 0
    assert result.manifest["raw"]["sha256"] == raw_digest
    assert result.manifest["environment"]["python"].startswith("3.")
    for name in [
        "validation_raw.json",
        "validation_processed.json",
        "transformations.json",
        "validation_report.md",
    ]:
        assert (paths.reports / name).exists()
    assert json.loads((paths.reports / "validation_raw.json").read_text())["passed"] is True

    run_data_pipeline(data_config, paths)  # a rerun leaves the raw file untouched
    assert sha256_file(result.raw_path) == raw_digest


def test_processed_parquet_round_trips_categoricals(data_config, paths):
    run_data_pipeline(data_config, paths)
    processed = pd.read_parquet(paths.processed / PROCESSED_FILE)
    assert isinstance(processed["segment"].dtype, pd.CategoricalDtype)
    assert processed["history_segment"].cat.ordered


def test_failed_validation_stops_before_processing(data_config, paths, source_file, raw_frame):
    path, _ = source_file
    raw_frame.loc[2, "spend"] = ""
    digest = write_csv(raw_frame, path)
    source = dataclasses.replace(data_config.sources[0], sha256=digest)
    config = dataclasses.replace(data_config, sources=[source])
    with pytest.raises(ValidationError, match="missing_values"):
        run_data_pipeline(config, paths)
    assert not (paths.processed / PROCESSED_FILE).exists()
    assert (paths.reports / "validation_raw.json").exists()


def test_load_processed_requires_the_pipeline_to_have_run(paths):
    with pytest.raises(FileNotFoundError, match="make data"):
        load_processed(paths)
