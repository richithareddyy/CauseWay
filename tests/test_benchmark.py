"""Benchmark construction: seeds, rejection sampling, estimand-matched targets and Monte Carlo metrics."""

import copy
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy import special

import causeway.config as config
from causeway.benchmark import (
    OverlapGuardError,
    _hajek_difference,
    acceptance_probabilities,
    build_pair,
    calibrate_intercept,
    estimator_seed,
    monte_carlo_summary,
    naive_bias_decomposition,
    oracle_ipw,
    overlap_guard,
    run_construction_study,
    selection_uniforms,
)
from causeway.config import FrozenSpecificationError, load_benchmark_config
from causeway.processing import TransformationLog, build_interim, build_processed
from conftest import make_raw_frame


@pytest.fixture(scope="module")
def spec():
    return load_benchmark_config(1)


@pytest.fixture(scope="module")
def processed():
    log = TransformationLog()
    return build_processed(build_interim(make_raw_frame(n_per_arm=1500, seed=3), log), log)


def test_seeds_are_deterministic_and_distinct():
    a = selection_uniforms(20080320, 0, 5, 100)
    np.testing.assert_array_equal(a, selection_uniforms(20080320, 0, 5, 100))
    assert not np.array_equal(a, selection_uniforms(20080320, 0, 6, 100))
    assert not np.array_equal(a, selection_uniforms(20080320, 1, 5, 100))
    assert estimator_seed(20080320, 0, 5) != estimator_seed(20080320, 0, 6)


def test_intercept_calibration_hits_the_target_mean():
    index = np.random.default_rng(0).normal(size=5000)
    alpha = calibrate_intercept(index, 2.0, 0.5, 1e-10)
    assert special.expit(alpha + 2.0 * index).mean() == pytest.approx(0.5, abs=1e-8)


def test_acceptance_probabilities_are_valid():
    rng = np.random.default_rng(1)
    e_star, treated = rng.uniform(0.05, 0.95, 1000), rng.random(1000) < 0.5
    acceptance, m = acceptance_probabilities(e_star, treated)
    assert acceptance.max() == pytest.approx(1.0) and acceptance.min() > 0
    assert m >= 1.0


def test_rejection_sampling_makes_propensity_equal_design_and_preserves_covariates():
    rng = np.random.default_rng(2)
    n = 200_000
    x = rng.normal(size=n)
    treated = rng.random(n) < 0.5  # randomized trial
    e_star = special.expit(-0.2 + 1.0 * x)
    acceptance, _ = acceptance_probabilities(e_star, treated)
    keep = rng.random(n) <= acceptance
    bins = np.digitize(x, [-1.0, 0.0, 1.0])
    for b in range(4):
        in_bin = keep & (bins == b)
        assert treated[in_bin].mean() == pytest.approx(e_star[in_bin].mean(), abs=0.01)
    assert x[keep].mean() == pytest.approx(x.mean(), abs=0.01)
    assert x[keep].std() == pytest.approx(x.std(), abs=0.01)


def test_oracle_ipw_recovers_the_full_trial_difference_with_heterogeneous_effects():
    rng = np.random.default_rng(4)
    n = 40_000
    x = rng.normal(size=n)
    treated = rng.random(n) < 0.5
    y = 1.0 + x + treated * (1.0 + 0.8 * x) + rng.normal(size=n)  # effect varies with x
    e_star = special.expit(1.0 * x)
    acceptance, _ = acceptance_probabilities(e_star, treated)
    benchmark = y[treated].mean() - y[~treated].mean()
    errors, naive = [], []
    for _ in range(60):
        keep = rng.random(n) <= acceptance
        errors.append(oracle_ipw(y[keep], treated[keep], e_star[keep], 0.95)["estimate"] - benchmark)
        naive.append(y[keep & treated].mean() - y[keep & ~treated].mean() - benchmark)
    assert abs(np.mean(errors)) < 3 * np.std(errors) / np.sqrt(len(errors))
    assert np.mean(naive) > 10 * np.std(naive) / np.sqrt(len(naive))  # the selection really confounds


def test_hajek_difference_reduces_to_difference_in_means_and_weights_toward_att():
    y = np.array([4.0, 6.0, 1.0, 3.0])
    treated = np.array([True, True, False, False])
    assert _hajek_difference(y, treated, np.ones(4))[0] == pytest.approx(5.0 - 2.0)
    weights = np.array([3.0, 1.0, 3.0, 1.0])
    assert _hajek_difference(y, treated, weights)[0] == pytest.approx((12 + 6) / 4 - (3 + 3) / 4)


def test_pair_designs_share_selection_between_a_and_c(processed, spec):
    pair = build_pair(processed, spec["comparisons"][0], spec)
    assert set(pair.designs) == {"A", "B", "C"}
    np.testing.assert_allclose(pair.designs["A"].e_star, pair.designs["C"].e_star)
    for scenario in ("A", "B"):
        assert pair.designs[scenario].e_star.mean() == pytest.approx(0.5, abs=1e-8)
    np.testing.assert_array_equal(pair.retained("A", 3, 20080320), pair.retained("C", 3, 20080320))
    assert "recency" in pair.analyst_columns("A", spec)
    assert "recency" not in pair.analyst_columns("C", spec)
    assert pair.analyst_columns("C", spec) == [c for c in pair.analyst_columns("A", spec) if c != "recency"]


def test_pair_contains_only_its_two_arms_in_row_order(processed, spec):
    pair = build_pair(processed, spec["comparisons"][1], spec)
    assert set(pair.frame["segment"].astype(str)) == {"Womens E-Mail", "No E-Mail"}
    assert pair.frame["row_id"].is_monotonic_increasing
    assert pair.treated.sum() == (pair.frame["segment"] == "Womens E-Mail").sum()


def test_overlap_guard_raises_when_violated(processed, spec):
    strong = copy.deepcopy(spec)
    strong["scenarios"]["A"]["selection_strength_k"] = 8.0
    with pytest.raises(OverlapGuardError, match="amendment"):
        overlap_guard(build_pair(processed, strong["comparisons"][0], strong), strong)


def test_naive_bias_decomposition_is_an_identity(processed, spec):
    table = naive_bias_decomposition(build_pair(processed, spec["comparisons"][0], spec), spec)
    np.testing.assert_allclose(
        table["bias_vs_ate"], table["baseline_bias"] + table["heterogeneity_term"], atol=1e-12
    )


def test_construction_study_structure(processed, spec):
    study = run_construction_study(processed, spec, n_draws=3)
    assert len(study.draws) == 2 * 2 * 3  # comparisons x scenarios A, B x draws
    assert set(study.estimates["estimator"]) == {"naive_difference", "oracle_ipw"}
    assert study.estimates["estimand"].notna().all() and (study.estimates["benchmark"] == "ATE").all()
    assert set(study.benchmarks["estimand"]) >= {"ATE", "ATT", "ATC"}
    assert study.guard.loc[study.guard["scenario"] == "A", "passed"].all()


def test_monte_carlo_summary_metrics():
    estimates = pd.DataFrame(
        {
            "comparison": "c",
            "scenario": "A",
            "outcome": "y",
            "estimator": "e",
            "estimand": "ATE",
            "benchmark": "ATE",
            "estimate": [1.0, 3.0, np.nan],
            "se": [1.0, 1.0, 1.0],
            "ci_low": [0.0, 2.5, np.nan],
            "ci_high": [2.0, 3.5, np.nan],
        }
    )
    benchmarks = pd.DataFrame(
        {"comparison": ["c"], "scenario": ["A"], "outcome": ["y"], "estimand": ["ATE"], "value": [1.5]}
    )
    row = monte_carlo_summary(estimates, benchmarks).iloc[0]
    assert row["bias"] == pytest.approx(0.5)
    assert row["absolute_bias"] == pytest.approx(1.0)
    assert row["rmse"] == pytest.approx(np.sqrt((0.25 + 2.25) / 2))
    assert row["coverage"] == pytest.approx(0.5)
    assert row["failure_rate"] == pytest.approx(1 / 3)


def test_modified_frozen_specification_is_refused(tmp_path, monkeypatch):
    shutil.copytree(config.CONFIG_DIR, tmp_path / "configs")
    path = tmp_path / "configs" / "benchmark" / "v1.toml"
    path.write_text(path.read_text().replace("selection_strength_k = 2.0", "selection_strength_k = 1.5"))
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "configs")
    with pytest.raises(FrozenSpecificationError, match="create a new version"):
        load_benchmark_config(1)
