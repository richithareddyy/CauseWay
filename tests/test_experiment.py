"""Part I estimators, checked against hand calculations, established libraries and known effects."""

import itertools
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from scipy.stats._resampling import _bca_interval  # private, pinned scipy version; used only as a reference

from causeway import eda
from causeway.experiment import (
    ESTIMAND,
    analyze_experiment,
    assignment_diagnostics,
    bca_interval,
    binary_contrast,
    bootstrap_means,
    dim_acceleration,
    holm,
    lin_adjusted,
    load_experiment_plan,
    neyman_difference,
    percentile_interval,
    randomization_test,
    stream_rng,
)
from causeway.features import covariate_matrix
from causeway.processing import TransformationLog, build_interim, build_processed


@pytest.fixture
def experiment():
    """Completely randomized experiment with a known constant effect of 2 and a prognostic covariate."""
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.normal(size=n)
    t = np.zeros(n, dtype=bool)
    t[rng.choice(n, n // 2, replace=False)] = True
    y = 1.0 + 3.0 * x + 2.0 * t + rng.normal(size=n)
    return y, t, pd.DataFrame({"x": x})


def test_neyman_matches_hc2_regression(experiment):
    y, t, _ = experiment
    estimate = neyman_difference(y[t], y[~t])
    fit = sm.OLS(y, sm.add_constant(t.astype(float))).fit(cov_type="HC2")
    assert estimate.estimate == pytest.approx(fit.params[1])
    assert estimate.se == pytest.approx(fit.bse[1])
    assert estimate.ci_low < 2.0 < estimate.ci_high


def test_neyman_hand_calculation():
    estimate = neyman_difference(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 3.0]))
    assert estimate.estimate == pytest.approx(1.0)
    assert estimate.se == pytest.approx(math.sqrt(1 / 3 + 3 / 3))


def test_lin_adjustment_recovers_effect_with_smaller_se(experiment):
    y, t, x = experiment
    lin = lin_adjusted(y, t, x)
    neyman = neyman_difference(y[t], y[~t])
    assert abs(lin.estimate - 2.0) < 4 * lin.se
    assert lin.se < 0.5 * neyman.se  # x explains most of the outcome variance


def test_lin_with_uninformative_covariate_is_close_to_neyman(experiment):
    y, t, _ = experiment
    noise = pd.DataFrame({"z": np.random.default_rng(3).normal(size=len(y))})
    lin, neyman = lin_adjusted(y, t, noise), neyman_difference(y[t], y[~t])
    assert lin.estimate == pytest.approx(neyman.estimate, abs=0.05)
    assert lin.se == pytest.approx(neyman.se, rel=0.02)


def test_randomization_test_matches_exhaustive_enumeration():
    y_t, y_c = np.array([3.1, 4.0, 2.2, 5.0]), np.array([1.0, 2.5, 0.4, 3.0])
    pooled = np.concatenate([y_t, y_c])
    observed = abs(y_t.mean() - y_c.mean())
    stats_all = [
        abs(pooled[list(s)].mean() - np.delete(pooled, list(s)).mean())
        for s in itertools.combinations(range(8), 4)
    ]
    exact = np.mean([s >= observed - 1e-12 for s in stats_all])
    p = randomization_test(y_t, y_c, 20000, np.random.default_rng(0))[0]
    assert p == pytest.approx(exact, abs=0.01)


def test_randomization_test_floor_and_null_behaviour(experiment):
    y, t, _ = experiment
    assert randomization_test(y[t], y[~t], 500, np.random.default_rng(1))[0] == pytest.approx(1 / 501)
    null_y = np.random.default_rng(2).normal(size=400)
    assert randomization_test(null_y[:200], null_y[200:], 2000, np.random.default_rng(1))[0] > 0.01


def test_randomization_test_handles_several_outcomes_and_is_reproducible(experiment):
    y, t, _ = experiment
    outcomes = np.column_stack([y, -y, np.zeros_like(y)])
    p1 = randomization_test(outcomes[t], outcomes[~t], 300, stream_rng(5, 1001))
    p2 = randomization_test(outcomes[t], outcomes[~t], 300, stream_rng(5, 1001))
    np.testing.assert_array_equal(p1, p2)
    assert p1[0] == p1[1] and p1[2] == 1.0


def test_bootstrap_is_reproducible_and_seed_sensitive(experiment):
    y, t, _ = experiment
    a = bootstrap_means([y[t], y[~t]], 200, stream_rng(20080320, 1000))
    b = bootstrap_means([y[t], y[~t]], 200, stream_rng(20080320, 1000))
    c = bootstrap_means([y[t], y[~t]], 200, stream_rng(20080320, 1001))
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])
    assert a[0].shape == (200, 1)


def test_bootstrap_se_matches_neyman_se(experiment):
    y, t, _ = experiment
    means = bootstrap_means([y[t], y[~t]], 4000, np.random.default_rng(9))
    draws = (means[0] - means[1])[:, 0]
    assert draws.std(ddof=1) == pytest.approx(neyman_difference(y[t], y[~t]).se, rel=0.06)


def test_bca_matches_scipy_reference_on_identical_draws():
    rng = np.random.default_rng(4)
    y_t, y_c = rng.exponential(2.0, 60), rng.exponential(1.0, 50)
    means = bootstrap_means([y_t, y_c], 3000, rng)
    draws = (means[0] - means[1])[:, 0]
    estimate = y_t.mean() - y_c.mean()

    def statistic(a, b, axis=-1):
        return np.mean(a, axis=axis) - np.mean(b, axis=axis)

    alpha_1, alpha_2, a_hat = _bca_interval(
        (y_t, y_c), statistic, axis=-1, alpha=0.025, theta_hat_b=draws, batch=None
    )
    assert dim_acceleration(y_t, y_c) == pytest.approx(float(a_hat))
    expected = np.percentile(draws, [100 * float(alpha_1), 100 * float(alpha_2)])
    np.testing.assert_allclose(bca_interval(draws, estimate, float(a_hat)), expected)


def test_bca_reduces_to_percentile_for_symmetric_unbiased_draws():
    draws = np.random.default_rng(0).normal(size=20001)
    draws = np.concatenate([draws, -draws])  # exactly symmetric about 0
    np.testing.assert_allclose(bca_interval(draws, 0.0, 0.0), percentile_interval(draws), atol=1e-9)


def test_binary_contrast_matches_wilson_based_newcombe():
    y_t = np.r_[np.ones(30), np.zeros(170)].astype(int)
    y_c = np.r_[np.ones(15), np.zeros(185)].astype(int)
    result = binary_contrast(y_t, y_c)

    z = stats.norm.ppf(0.975)

    def wilson(k, n):
        p = k / n
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return centre - half, centre + half

    (l1, u1), (l0, u0) = wilson(30, 200), wilson(15, 200)
    d = 30 / 200 - 15 / 200
    assert result["rd_ci_low"] == pytest.approx(d - math.sqrt((30 / 200 - l1) ** 2 + (u0 - 15 / 200) ** 2))
    assert result["rd_ci_high"] == pytest.approx(d + math.sqrt((u1 - 30 / 200) ** 2 + (15 / 200 - l0) ** 2))
    assert result["risk_ratio"] == pytest.approx(2.0)
    assert result["rr_ci_low"] < 2.0 < result["rr_ci_high"]


def test_holm_hand_calculation():
    assert holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm([0.03, 0.02]) == pytest.approx([0.04, 0.04])


def test_assignment_diagnostics_detect_covariate_dependent_assignment():
    rng = np.random.default_rng(8)
    x = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(size=3000)})
    random_t = rng.random(3000) < 0.5
    selected_t = rng.random(3000) < 1 / (1 + np.exp(-2 * x["a"].to_numpy()))
    random_d = assignment_diagnostics(x, random_t, folds=5, seed=1)
    selected_d = assignment_diagnostics(x, selected_t, folds=5, seed=1)
    assert random_d["lr_p_value"] > 0.001 and abs(random_d["out_of_fold_auc"] - 0.5) < 0.05
    assert selected_d["lr_p_value"] < 1e-10 and selected_d["out_of_fold_auc"] > 0.75


@pytest.fixture
def processed(raw_frame):
    log = TransformationLog()
    return build_processed(build_interim(raw_frame, log), log)


@pytest.fixture
def small_plan():
    plan = load_experiment_plan()
    plan["inference"] = {**plan["inference"], "bootstrap_resamples": 200, "permutations": 200}
    plan["diagnostics"] = {**plan["diagnostics"], "classifier_folds": 3}
    return plan


def test_analyze_experiment_structure_and_estimands(processed, small_plan):
    results = analyze_experiment(processed, small_plan)
    effects = results.effects
    assert len(effects) == 3 * 3 * 4
    assert (effects["estimand"] == ESTIMAND).all()
    holm_rows = effects["p_holm"].notna()
    assert set(effects.loc[holm_rows, "role"]) == {"co-primary"}
    assert set(effects.loc[holm_rows, "estimator"]) == {"difference_in_means"}
    assert results.bootstrap_draws.shape == (200, 9)
    assert set(results.binary["outcome"]) == {"visit", "conversion"}


def test_analyze_experiment_matches_direct_neyman(processed, small_plan):
    effects = analyze_experiment(processed, small_plan).effects
    row = effects[
        (effects["contrast"] == "mens_vs_control")
        & (effects["outcome"] == "spend")
        & (effects["estimator"] == "difference_in_means")
    ].iloc[0]
    y = processed.set_index("segment")["spend"]
    direct = neyman_difference(y.loc["Mens E-Mail"].to_numpy(), y.loc["No E-Mail"].to_numpy())
    assert row["estimate"] == pytest.approx(direct.estimate)
    assert row["se"] == pytest.approx(direct.se)


def test_analysis_is_reproducible(processed, small_plan):
    a = analyze_experiment(processed, small_plan)
    b = analyze_experiment(processed, small_plan)
    pd.testing.assert_frame_equal(a.effects, b.effects)
    pd.testing.assert_frame_equal(a.bootstrap_draws, b.bootstrap_draws)


def test_regression_adjustment_uses_only_pretreatment_covariates(small_plan, processed):
    columns = small_plan["regression_adjustment"]["covariates"]
    assert set(columns) <= set(covariate_matrix(processed).columns)
    assert not {"spend", "visit", "conversion", "segment", "arm"} & set(columns)


def test_spend_boundary_summary(processed):
    summary = eda.spend_boundary(processed, [[100.0, 200.0]])
    positive = processed.loc[processed["spend"] > 0, "spend"]
    assert summary["maximum"] == pytest.approx(positive.max())
    assert summary["n_converters"] == len(positive)
    assert sum(v["n_converters"] for v in summary["converters_at_maximum_by_arm"].values()) == len(positive)


def test_ratios_are_undefined_rather_than_infinite_when_an_arm_has_no_events():
    result = binary_contrast(np.r_[np.ones(3), np.zeros(97)].astype(int), np.zeros(100, dtype=int))
    assert np.isnan(result["risk_ratio"]) and np.isnan(result["rr_ci_low"])
    assert result["risk_difference"] == pytest.approx(0.03)
    assert np.isfinite(result["rd_ci_low"]) and np.isfinite(result["rd_ci_high"])
