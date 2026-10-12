"""Robustness checks and sensitivity analysis."""

import numpy as np
import pandas as pd
import pytest

from causeway import evaluation as ev
from causeway import schema
from causeway.processing import TransformationLog, build_interim, build_processed
from causeway.robustness import experiment_robustness, spend_versions
from causeway.sensitivity import (
    benchmark_bounds,
    bias_bound,
    confounder_strength,
    e_value,
    ovb_analysis,
    robustness_value,
    rosenbaum_gamma,
)
from conftest import make_raw_frame


@pytest.fixture(scope="module")
def processed():
    log = TransformationLog()
    return build_processed(build_interim(make_raw_frame(n_per_arm=1500, seed=13), log), log)


def test_spend_versions_keep_every_customer_and_only_touch_the_top(processed):
    versions = spend_versions(processed)
    top = processed["spend"].max()
    for name, values in versions.items():
        assert len(values) == len(processed), name
    recoded = next(v for name, v in versions.items() if "recoded" in name)
    assert recoded.max() < top
    changed = recoded != processed["spend"]
    assert (processed.loc[changed, "spend"] == top).all()


def test_experiment_robustness_drops_only_exact_duplicates(processed):
    table = experiment_robustness(processed)
    full = table[table["analysis"] == "spend version"].groupby("contrast")["n"].first()
    deduplicated = table[table["analysis"].str.startswith("without duplicate rows")]
    n_unique = len(processed.drop_duplicates(subset=schema.COLUMNS))
    assert f"n = {n_unique:,}" in deduplicated["analysis"].iloc[0]
    assert (deduplicated.groupby("contrast")["n"].first() <= full).all()
    assert set(deduplicated["outcome"]) == {"spend", "visit", "conversion"}


def test_robustness_value_formula():
    # f = |t| / sqrt(dof) = 1 gives RV = (sqrt(5) - 1) / 2
    assert robustness_value(10.0, 100.0) == pytest.approx((np.sqrt(5) - 1) / 2)
    assert robustness_value(1.0, 100.0, alpha=0.05) == 0.0  # not significant to begin with


def test_bias_bound_is_zero_without_confounding_and_grows_with_strength():
    assert bias_bound(1.0, 100, 0.0, 0.3) == 0.0
    assert bias_bound(1.0, 100, 0.2, 0.2) > bias_bound(1.0, 100, 0.1, 0.1)
    r2_yz, r2_dz = benchmark_bounds(0.1, 0.1, 1.0)
    assert 0 < r2_dz < 1 and 0 < r2_yz <= 1


def test_omitted_variable_bound_matches_the_actual_omitted_variable_bias():
    rng = np.random.default_rng(1)
    n = 20_000
    x = rng.normal(size=n)
    z = rng.normal(size=n)  # the confounder we pretend not to see
    d = (0.5 * x + 0.8 * z + rng.normal(size=n) > 0).astype(float)
    y = 1.0 * d + x + 1.5 * z + rng.normal(size=n)
    covariates = pd.DataFrame({"x": x})
    summary, _ = ovb_analysis(y, d.astype(bool), covariates)
    strength = confounder_strength(y, d.astype(bool), covariates, z)
    implied = bias_bound(summary["se"], summary["dof"], strength["r2_yz"], strength["r2_dz"])
    assert implied == pytest.approx(summary["estimate"] - 1.0, rel=0.05)


def test_e_value_reference_values():
    assert e_value(2.0) == pytest.approx(2 + np.sqrt(2))
    assert e_value(0.5) == pytest.approx(e_value(2.0))
    assert e_value(1.0) == 1.0


def test_rosenbaum_gamma():
    assert np.isnan(rosenbaum_gamma(10, 10))  # no effect even without hidden bias
    strong, weak = rosenbaum_gamma(200, 50), rosenbaum_gamma(120, 80)
    assert strong > weak > 1


def test_robustness_methods_run_and_tag_their_specifications(processed):
    methods = [
        ev.propensity_specification_methods,
        ev.outcome_specification_methods,
        ev.trimming_methods,
        ev.caliper_methods,
        ev.refutation_methods,
        ev.sensitivity_methods,
    ]
    study = ev.run_estimator_study(processed, methods, scenarios=["A", "C"], n_draws=1)
    assert not [c for c in study.diagnostics.columns if c.endswith("_error")]
    specifications = set(study.estimates["specification"])
    assert {
        "propensity=alt_categorical",
        "outcome=alt_gbm",
        "trim=[0.05,0.95]",
        "caliper=none",
        "placebo",
    } <= specifications
    assert set(study.estimates["outcome"]) == {"spend", "visit"}
    c = study.diagnostics[study.diagnostics["scenario"] == "C"].iloc[0]
    assert np.isfinite(c["visit_hidden_recency_r2_dz"])


def test_produces_declares_estimators_for_failure_rows():
    assert ev.trimming_methods.estimators[0][0] == "ipw_ate"
    assert ev.sensitivity_methods.estimators == []
