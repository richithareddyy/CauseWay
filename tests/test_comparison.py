"""Milestone 7 comparison: provenance checks, completeness, estimand alignment and table construction."""

import json

import numpy as np
import pandas as pd
import pytest

from causeway.comparison import (
    ASSUMPTIONS,
    ESTIMATOR_ORDER,
    REFERENCE_ESTIMATORS,
    ProvenanceMismatch,
    check_provenance,
    display_table,
    load_benchmark_estimates,
    performance_table,
)
from causeway.config import DataPaths
from causeway.provenance import PROVENANCE_FILE

BENCHMARK_OF = {
    "naive_difference": "ATE",
    "oracle_ipw": "ATE",
    "outcome_regression": "ATE",
    "ipw_ate": "ATE",
    "aipw": "ATE",
    "dml_irm": "ATE",
    "ipw_att": "ATT",
    "matching_nn": "ATT",
}
ESTIMAND_OF = {**BENCHMARK_OF, "naive_difference": "unadjusted association"}
TRUTH = {"ATE": 1.0, "ATT": 1.5}


def make_estimates(estimators, n_draws=4, bias=None):
    rng = np.random.default_rng(0)
    bias = bias or {}
    rows = []
    for estimator in estimators:
        for draw in range(n_draws):
            value = TRUTH[BENCHMARK_OF[estimator]] + bias.get(estimator, 0.0) + rng.normal(0, 0.01)
            rows.append(
                {
                    "comparison": "mens_vs_control",
                    "scenario": "A",
                    "draw": draw,
                    "outcome": "spend",
                    "estimator": estimator,
                    "estimand": ESTIMAND_OF[estimator],
                    "benchmark": BENCHMARK_OF[estimator],
                    "estimate": value,
                    "se": 0.05,
                    "ci_low": value - 0.1,
                    "ci_high": value + 0.1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def benchmarks():
    return pd.DataFrame(
        [
            {
                "comparison": "mens_vs_control",
                "scenario": "A",
                "outcome": "spend",
                "estimand": e,
                "value": v,
                "se": 0.1,
            }
            for e, v in TRUTH.items()
        ]
    )


def write_source(root, name, estimates, provenance):
    directory = root / "reports" / "results" / name
    directory.mkdir(parents=True)
    estimates.to_csv(directory / "estimates.csv", index=False)
    (directory / PROVENANCE_FILE).write_text(json.dumps(provenance))


PROVENANCE = {
    "data": {"processed_sha256": "abc"},
    "benchmark": {"sha256": "d88c", "base_seed": 1, "n_draws": 4},
}


def test_load_combines_sources_and_checks_shared_references(tmp_path):
    propensity = make_estimates(["naive_difference", "oracle_ipw", "ipw_ate", "ipw_att", "matching_nn"])
    estimation = make_estimates(["naive_difference", "oracle_ipw", "outcome_regression", "aipw", "dml_irm"])
    write_source(tmp_path, "propensity", propensity, PROVENANCE)
    write_source(tmp_path, "estimation", estimation, PROVENANCE)
    combined, _ = load_benchmark_estimates(DataPaths(root=tmp_path), n_draws=4)
    assert set(combined["estimator"]) == set(ESTIMATOR_ORDER)
    assert len(combined) == len(ESTIMATOR_ORDER) * 4  # references are not duplicated


def test_load_refuses_mismatched_provenance(tmp_path):
    write_source(tmp_path, "propensity", make_estimates(["naive_difference"]), PROVENANCE)
    other = {**PROVENANCE, "benchmark": {**PROVENANCE["benchmark"], "base_seed": 2}}
    write_source(tmp_path, "estimation", make_estimates(["naive_difference"]), other)
    with pytest.raises(ProvenanceMismatch, match="base_seed"):
        load_benchmark_estimates(DataPaths(root=tmp_path), n_draws=4)


def test_load_refuses_references_that_differ_between_runs(tmp_path):
    propensity = make_estimates(["naive_difference", "oracle_ipw", "ipw_ate", "ipw_att", "matching_nn"])
    estimation = make_estimates(["naive_difference", "oracle_ipw", "outcome_regression", "aipw", "dml_irm"])
    estimation.loc[estimation["estimator"] == "oracle_ipw", "estimate"] += 1e-6
    write_source(tmp_path, "propensity", propensity, PROVENANCE)
    write_source(tmp_path, "estimation", estimation, PROVENANCE)
    with pytest.raises(ProvenanceMismatch, match="reference estimators differ"):
        load_benchmark_estimates(DataPaths(root=tmp_path), n_draws=4)


def test_load_refuses_incomplete_draws(tmp_path):
    propensity = make_estimates(["naive_difference", "oracle_ipw", "ipw_ate", "ipw_att", "matching_nn"])
    estimation = make_estimates(["naive_difference", "oracle_ipw", "outcome_regression", "aipw", "dml_irm"])
    estimation = estimation[~((estimation["estimator"] == "aipw") & (estimation["draw"] == 3))]
    write_source(tmp_path, "propensity", propensity, PROVENANCE)
    write_source(tmp_path, "estimation", estimation, PROVENANCE)
    with pytest.raises(ProvenanceMismatch, match="incomplete"):
        load_benchmark_estimates(DataPaths(root=tmp_path), n_draws=4)


def test_check_provenance_accepts_matching_records():
    assert check_provenance({"a": PROVENANCE, "b": dict(PROVENANCE)}) == PROVENANCE


def test_performance_table_aligns_estimands_and_flags_bias(benchmarks):
    estimates = make_estimates(
        ESTIMATOR_ORDER, n_draws=50, bias={"naive_difference": 0.5, "matching_nn": -0.3}
    )
    table = performance_table(estimates, benchmarks).set_index("estimator")
    assert table.loc["ipw_att", "benchmark_value"] == TRUTH["ATT"]
    assert table.loc["matching_nn", "benchmark_value"] == TRUTH["ATT"]
    assert table.loc["aipw", "benchmark_value"] == TRUTH["ATE"]
    assert not table.loc["naive_difference", "unbiased_within_mc_error"]
    assert not table.loc["matching_nn", "unbiased_within_mc_error"]
    assert table.loc["matching_nn", "bias_pct"] == pytest.approx(-0.2, abs=0.01)
    assert (table["coverage_minus_oracle"] == table["coverage"] - table.loc["oracle_ipw", "coverage"]).all()
    assert list(table.index) == ESTIMATOR_ORDER
    assert table["key_assumptions"].notna().all()


def test_display_table_has_experiment_rows_and_matched_benchmarks(benchmarks):
    estimates = make_estimates(ESTIMATOR_ORDER, n_draws=2)
    experiment = pd.DataFrame(
        [
            {
                "contrast": "mens_vs_control",
                "outcome": "spend",
                "estimator": "difference_in_means",
                "estimate": 1.0,
                "ci_low": 0.8,
                "ci_high": 1.2,
            }
        ]
    )
    table = display_table(estimates, benchmarks, experiment, draw=1)
    assert list(table["estimator"][:2]) == ["experiment", "experiment_att"]
    assert len(table) == 2 + len(ESTIMATOR_ORDER)
    rows = table.set_index("estimator")
    assert rows.loc["matching_nn", "benchmark_value"] == TRUTH["ATT"]
    assert rows.loc["dml_irm", "benchmark_value"] == TRUTH["ATE"]
    assert "estimand" in table.columns
    assert rows.loc["experiment", "ci_covers_benchmark"]


def test_every_estimator_has_documented_assumptions():
    assert set(ESTIMATOR_ORDER) <= set(ASSUMPTIONS)
    assert set(REFERENCE_ESTIMATORS) <= set(ESTIMATOR_ORDER)
