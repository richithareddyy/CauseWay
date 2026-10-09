"""Raw-data validation checks: each one must catch its defect and none may alter the data."""

import re
import warnings

import pandas as pd
import pytest

from causeway import schema
from causeway.validation import validate_raw


def status(report, name):
    return next(check for check in report.checks if check.name == name).status


def test_valid_fixture_passes(raw_frame):
    report = validate_raw(raw_frame)
    assert report.passed, [c.message for c in report.failures()]
    assert status(report, "schema_columns") == "pass"
    assert status(report, "binary_encoding") == "pass"


def test_validation_does_not_modify_its_input(raw_frame):
    before = raw_frame.copy()
    validate_raw(raw_frame)
    pd.testing.assert_frame_equal(raw_frame, before)


def test_expected_row_count_is_enforced(raw_frame):
    assert status(validate_raw(raw_frame, expected_rows=len(raw_frame)), "row_count") == "pass"
    assert status(validate_raw(raw_frame, expected_rows=64000), "row_count") == "fail"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f.drop(columns="spend"),
        lambda f: f.assign(extra="1"),
        lambda f: f[list(reversed(schema.COLUMNS))],
    ],
    ids=["missing_column", "unexpected_column", "wrong_order"],
)
def test_schema_violations_fail_and_stop_further_checks(raw_frame, mutate):
    report = validate_raw(mutate(raw_frame))
    assert status(report, "schema_columns") == "fail"
    assert len(report.checks) == 1


@pytest.mark.parametrize("token", ["", "NA", "NaN", "null", " "])
def test_missing_values_fail_and_are_not_imputed(raw_frame, token):
    raw_frame.loc[3, "history"] = token
    report = validate_raw(raw_frame)
    assert status(report, "missing_values") == "fail"
    assert report.checks[-1].details["by_column"] == {"history": 1}


@pytest.mark.parametrize(
    ("column", "value", "check"),
    [
        ("recency", "1.5", "data_types"),
        ("history", "abc", "data_types"),
        ("spend", "inf", "data_types"),
        ("segment", "SMS", "domain_segment"),
        ("channel", "Mail", "domain_channel"),
        ("zip_code", "Suburbs", "domain_zip_code"),
        ("history_segment", "8) $5,000 +", "domain_history_segment"),
        ("visit", "2", "binary_encoding"),
        ("conversion", "-1", "binary_encoding"),
        ("recency", "13", "recency_range"),
        ("recency", "0", "recency_range"),
        ("history", "-5.00", "history_positive"),
    ],
)
def test_invalid_values_are_detected(raw_frame, column, value, check):
    raw_frame.loc[5, column] = value
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a defect must be reported, never trip numeric warnings downstream
        report = validate_raw(raw_frame)
    assert status(report, check) == "fail"
    assert not report.passed


def test_negative_spend_fails(raw_frame):
    row = raw_frame.index[raw_frame["conversion"] == "1"][0]
    raw_frame.loc[row, "spend"] = "-10.00"
    assert status(validate_raw(raw_frame), "spend_nonnegative") == "fail"


def test_history_outside_its_segment_bin_fails(raw_frame):
    raw_frame.loc[0, ["history_segment", "history"]] = ["1) $0 - $100", "250.00"]
    assert status(validate_raw(raw_frame), "history_segment_consistency") == "fail"


def test_spend_without_conversion_fails(raw_frame):
    row = raw_frame.index[raw_frame["conversion"] == "0"][0]
    raw_frame.loc[row, "spend"] = "45.00"
    assert status(validate_raw(raw_frame), "spend_conversion_consistency") == "fail"


def test_conversion_without_spend_fails(raw_frame):
    row = raw_frame.index[raw_frame["conversion"] == "1"][0]
    raw_frame.loc[row, "spend"] = "0"
    assert status(validate_raw(raw_frame), "spend_conversion_consistency") == "fail"


def test_conversion_without_visit_is_a_warning(raw_frame):
    row = raw_frame.index[raw_frame["conversion"] == "1"][0]
    raw_frame.loc[row, "visit"] = "0"
    assert status(validate_raw(raw_frame), "conversion_implies_visit") == "warn"


def test_missing_treatment_arm_fails(raw_frame):
    raw_frame.loc[raw_frame["segment"] == "Womens E-Mail", "segment"] = "Mens E-Mail"
    assert status(validate_raw(raw_frame), "treatment_arms") == "fail"


def test_severe_allocation_imbalance_warns(raw_frame):
    extra = raw_frame[raw_frame["segment"] == "Mens E-Mail"]
    imbalanced = pd.concat([raw_frame] + [extra] * 6, ignore_index=True)
    assert status(validate_raw(imbalanced), "allocation_ratio") == "warn"


def test_covariate_imbalance_warns(raw_frame):
    raw_frame.loc[raw_frame["segment"] == "Mens E-Mail", "newbie"] = "1"
    raw_frame.loc[raw_frame["segment"] == "No E-Mail", "newbie"] = "0"
    assert status(validate_raw(raw_frame), "balance_mens_vs_control") == "warn"


def test_duplicates_are_reported_but_kept(raw_frame):
    report = validate_raw(raw_frame)
    check = next(c for c in report.checks if c.name == "exact_duplicate_rows")
    assert check.status == "info"
    assert check.details["extra_copies"] == 1
    assert report.n_rows == len(raw_frame)


def test_surburban_spelling_is_reported_for_normalisation(raw_frame):
    report = validate_raw(raw_frame)
    check = next(c for c in report.checks if c.name == "category_spelling")
    assert check.details["variants"] == {"Surburban": int((raw_frame["zip_code"] == "Surburban").sum())}


def test_spend_boundary_is_recorded_as_an_anomaly_not_a_cap(raw_frame):
    check = next(c for c in validate_raw(raw_frame).checks if c.name == "spend_upper_boundary")
    assert check.status == "info"
    assert "unknown" in check.message and "cap" not in check.message


def test_markdown_report_escapes_table_delimiters(raw_frame):
    markdown = validate_raw(raw_frame).to_markdown()
    balance_line = next(line for line in markdown.splitlines() if "balance_mens_vs_control" in line)
    assert "max \\|SMD\\|" in balance_line
    assert len(re.findall(r"(?<!\\)\|", balance_line)) == 5  # four cells -> five unescaped delimiters


def test_reordered_columns_are_described_as_an_order_problem(raw_frame):
    report = validate_raw(raw_frame[list(reversed(schema.COLUMNS))])
    assert report.checks[0].message == "all expected columns present but in a different order"
