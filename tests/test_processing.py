"""Processing transformations: every change is explicit, logged, and loses no rows."""

import numpy as np
import pandas as pd
import pytest

from causeway import schema
from causeway.processing import (
    TransformationLog,
    build_interim,
    build_processed,
    normalize_zip_code,
    recency_band,
)


@pytest.fixture
def processed_and_log(raw_frame):
    log = TransformationLog()
    interim = build_interim(raw_frame, log)
    return interim, build_processed(interim, log), log


def test_interim_changes_types_but_not_values(raw_frame):
    interim = build_interim(raw_frame, TransformationLog())
    assert interim["recency"].dtype == "int64" and interim["spend"].dtype == "float64"
    assert (interim["zip_code"] == raw_frame["zip_code"]).all()  # labels untouched at this stage
    np.testing.assert_allclose(interim["history"], raw_frame["history"].astype(float))


def test_no_rows_are_removed_and_row_id_traces_raw_order(raw_frame, processed_and_log):
    _, processed, _ = processed_and_log
    assert len(processed) == len(raw_frame)
    assert processed["row_id"].tolist() == list(range(len(raw_frame)))
    assert processed.duplicated(subset=schema.COLUMNS).sum() == 1  # the fixture's duplicate survives


def test_surburban_is_normalised_and_logged_with_its_count(raw_frame, processed_and_log):
    _, processed, log = processed_and_log
    n_variant = int((raw_frame["zip_code"] == "Surburban").sum())
    assert n_variant > 0
    assert set(processed["zip_code"].astype(str)) <= {"Urban", "Suburban", "Rural"}
    assert (processed["zip_code"] == "Suburban").sum() == n_variant
    entry = next(e for e in log.entries if e.columns == ["zip_code"])
    assert entry.action == "recode 'Surburban' -> 'Suburban'" and entry.rows_affected == n_variant


def test_unknown_category_raises_instead_of_passing_through():
    with pytest.raises(ValueError, match="unrecognised zip_code"):
        normalize_zip_code(pd.Series(["Urban", "Downtown"]))


def test_categoricals_have_explicit_order(processed_and_log):
    _, processed, _ = processed_and_log
    assert list(processed["segment"].cat.categories) == schema.ARMS
    assert list(processed["channel"].cat.categories) == schema.CHANNELS
    assert processed["history_segment"].cat.ordered
    assert list(processed["arm"].cat.categories) == ["control", "mens", "womens"]


def test_arm_codes_follow_segment(processed_and_log):
    _, processed, _ = processed_and_log
    expected = processed["segment"].astype(str).map(schema.ARM_CODES)
    assert (processed["arm"].astype(str) == expected).all()


def test_derived_columns(processed_and_log):
    interim, processed, _ = processed_and_log
    np.testing.assert_allclose(processed["ln_history"], np.log(interim["history"]))


def test_recency_band_edges():
    bands = recency_band(pd.Series([1, 3, 4, 6, 7, 9, 10, 12]))
    assert bands.astype(str).tolist() == ["1-3", "1-3", "4-6", "4-6", "7-9", "7-9", "10-12", "10-12"]


def test_processing_does_not_mutate_the_interim_frame(raw_frame):
    log = TransformationLog()
    interim = build_interim(raw_frame, log)
    snapshot = interim.copy()
    build_processed(interim, log)
    pd.testing.assert_frame_equal(interim, snapshot)


def test_every_transformation_is_logged(processed_and_log):
    _, processed, log = processed_and_log
    logged_columns = {column for entry in log.entries for column in entry.columns}
    new_columns = set(processed.columns) - set(schema.COLUMNS)
    assert new_columns <= logged_columns
    assert [e.step for e in log.entries] == list(range(1, len(log.entries) + 1))
