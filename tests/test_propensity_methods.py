"""Propensity model, IPW and matching: known effects, hand calculations and a brute-force matching check."""

import numpy as np
import pandas as pd
import pytest
from scipy import special
from sklearn.metrics import roc_auc_score

from causeway.config import load_benchmark_config
from causeway.evaluation import (
    iter_samples,
    propensity_methods,
    reference_methods,
    run_estimator_study,
)
from causeway.matching import (
    greedy_match,
    match_on_propensity,
    matched_att,
    matching_weights,
    pooled_logit_sd,
)
from causeway.processing import TransformationLog, build_interim, build_processed
from causeway.propensity import fit_logistic_propensity, predictive_metrics, recovery_metrics
from causeway.weighting import (
    ate_weights,
    att_weights,
    hajek_difference,
    kish_ess,
    trim_mask,
    weight_diagnostics,
)
from conftest import make_raw_frame


@pytest.fixture(scope="module")
def confounded():
    """Observational data with known propensity and heterogeneous effect tau(x) = 1 + x."""
    rng = np.random.default_rng(21)
    n = 30_000
    x = rng.normal(size=n)
    e = special.expit(-0.3 + 1.2 * x)
    t = rng.random(n) < e
    y0 = 2.0 + 1.5 * x + rng.normal(size=n)
    y1 = y0 + 1.0 + x
    y = np.where(t, y1, y0)
    return {
        "x": pd.DataFrame({"x": x}),
        "t": t,
        "y": y,
        "e": e,
        "ate": float((y1 - y0).mean()),
        "att": float((y1 - y0)[t].mean()),
    }


def test_logistic_propensity_recovers_coefficients(confounded):
    fit = fit_logistic_propensity(confounded["x"], confounded["t"])
    assert fit.converged
    assert fit.coefficients["intercept"] == pytest.approx(-0.3, abs=0.05)
    assert fit.coefficients["x"] == pytest.approx(1.2, abs=0.05)
    assert recovery_metrics(fit.scores, confounded["e"])["rmse_vs_true_propensity"] < 0.01


def test_predictive_metrics_match_sklearn(confounded):
    metrics = predictive_metrics(confounded["e"], confounded["t"])
    assert metrics["auc"] == pytest.approx(roc_auc_score(confounded["t"], confounded["e"]))
    assert recovery_metrics(confounded["e"], confounded["e"])["rmse_vs_true_propensity"] == 0.0


def test_ipw_recovers_ate_and_att_where_naive_fails(confounded):
    t, y = confounded["t"], confounded["y"]
    scores = fit_logistic_propensity(confounded["x"], t).scores
    ate = hajek_difference(y, t, ate_weights(scores, t))
    att = hajek_difference(y, t, att_weights(scores, t))
    naive = y[t].mean() - y[~t].mean()
    assert abs(ate.estimate - confounded["ate"]) < 4 * ate.se
    assert abs(att.estimate - confounded["att"]) < 4 * att.se
    assert abs(naive - confounded["ate"]) > 20 * ate.se


def test_hajek_difference_hand_calculation():
    y = np.array([2.0, 4.0, 1.0, 3.0])
    t = np.array([True, True, False, False])
    w = np.array([1.0, 3.0, 1.0, 1.0])
    result = hajek_difference(y, t, w)
    assert result.estimate == pytest.approx((2 + 12) / 4 - 2.0)
    expected_var = ((1 * (2 - 3.5)) ** 2 + (3 * (4 - 3.5)) ** 2) / 16 + ((1 - 2) ** 2 + (3 - 2) ** 2) / 4
    assert result.se == pytest.approx(np.sqrt(expected_var))


def test_weights_definitions():
    scores, t = np.array([0.2, 0.5, 0.8]), np.array([True, False, False])
    np.testing.assert_allclose(ate_weights(scores, t), [5.0, 2.0, 5.0])
    np.testing.assert_allclose(att_weights(scores, t), [1.0, 1.0, 4.0])
    np.testing.assert_array_equal(trim_mask(scores, 0.3, 0.7), [False, True, False])


def test_kish_ess_and_weight_diagnostics():
    assert kish_ess(np.ones(50)) == pytest.approx(50)
    assert kish_ess(np.r_[np.ones(9), 91.0]) == pytest.approx(100**2 / (9 + 91**2))
    t = np.array([True, True, False, False])
    diagnostics = weight_diagnostics(np.array([1.0, 3.0, 2.0, 2.0]), t, "ate")
    assert diagnostics["ate_weight_max_treated"] == pytest.approx(1.5)
    assert diagnostics["ate_ess_share_control"] == pytest.approx(1.0)


def brute_force_match(t_logit, c_logit, order, c_rows, caliper):
    available = np.ones(len(c_logit), dtype=bool)
    out = np.full(len(t_logit), -1)
    for i in order:
        if not available.any():
            break
        candidates = np.flatnonzero(available)
        best = min(candidates, key=lambda j: (abs(c_logit[j] - t_logit[i]), c_rows[j]))
        if caliper is not None and abs(c_logit[best] - t_logit[i]) > caliper:
            continue
        out[i] = best
        available[best] = False
    return out


@pytest.mark.parametrize("caliper", [None, 0.05, 0.3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_greedy_match_equals_brute_force(seed, caliper):
    rng = np.random.default_rng(seed)
    t_logit, c_logit = rng.normal(0.5, 1, 60), rng.normal(0, 1, 45)
    order = np.argsort(-t_logit, kind="stable")
    c_rows = rng.permutation(45)
    result = greedy_match(t_logit, c_logit, order, c_rows, caliper)
    np.testing.assert_array_equal(
        result.control_for_treated, brute_force_match(t_logit, c_logit, order, c_rows, caliper)
    )
    used = result.control_for_treated[result.control_for_treated >= 0]
    assert len(used) == len(set(used.tolist()))  # without replacement
    if caliper is not None:
        assert (result.distances <= caliper).all()


def test_match_on_propensity_orders_treated_by_descending_score_and_applies_caliper():
    scores = np.array([0.9, 0.6, 0.55, 0.3, 0.2])
    t = np.array([True, True, False, False, False])
    rows = np.arange(5)
    result = match_on_propensity(scores, t, rows, caliper_sd=None)
    # 0.9 (highest) is matched first and takes the closest control (0.55), leaving 0.3 for 0.6.
    np.testing.assert_array_equal(result.control_for_treated, [0, 1])
    tight = match_on_propensity(scores, t, rows, caliper_sd=0.01)
    assert (tight.control_for_treated == -1).all()
    assert tight.caliper == pytest.approx(0.01 * pooled_logit_sd(scores, t))


def test_matched_att_and_weights():
    y = np.array([5.0, 7.0, 9.0, 1.0, 2.0, 3.0])
    t = np.array([True, True, True, False, False, False])
    result = match_on_propensity(np.array([0.8, 0.5, 0.2, 0.79, 0.5, 0.21]), t, np.arange(6), caliper_sd=None)
    att = matched_att(y, t, result)
    assert att["estimate"] == pytest.approx(((5 - 1) + (7 - 2) + (9 - 3)) / 3)
    assert att["share_treated_discarded"] == 0.0
    assert matching_weights(t, result).sum() == 6


@pytest.fixture(scope="module")
def processed():
    log = TransformationLog()
    return build_processed(build_interim(make_raw_frame(n_per_arm=1500, seed=5), log), log)


def test_estimator_study_rows_estimands_and_no_failures(processed):
    study = run_estimator_study(processed, [reference_methods, propensity_methods], n_draws=2)
    estimands = study.estimates.groupby("estimator")["benchmark"].first().to_dict()
    assert estimands == {
        "naive_difference": "ATE",
        "oracle_ipw": "ATE",
        "ipw_ate": "ATE",
        "ipw_att": "ATT",
        "matching_nn": "ATT",
    }
    assert study.estimates["estimate"].notna().all()
    assert len(study.diagnostics) == 2 * 3 * 2


def test_visible_balance_hides_the_hidden_covariate(processed):
    diagnostics = run_estimator_study(
        processed, [propensity_methods], scenarios=["A", "C"], n_draws=1
    ).diagnostics
    a, c = (diagnostics[diagnostics["scenario"] == s].iloc[0] for s in ("A", "C"))
    assert a["max_abs_smd_ipw_ate_visible"] == pytest.approx(a["max_abs_smd_ipw_ate"])
    assert c["max_abs_smd_ipw_ate_visible"] < c["max_abs_smd_ipw_ate"]


def test_analyst_inputs_never_contain_outcomes_or_hidden_covariates(processed):
    spec = load_benchmark_config(1)
    from causeway.benchmark import build_pair

    pairs = {c["name"]: build_pair(processed, c, spec) for c in spec["comparisons"]}
    for sample in iter_samples(pairs, spec, ["A", "C"], 1):
        assert not {"spend", "visit", "conversion", "segment", "arm"} & set(sample.x_analyst)
        assert ("recency" in sample.x_analyst) == (sample.scenario == "A")


def test_a_failing_method_is_recorded_as_failed_rows(processed, monkeypatch):
    def broken(sample, spec):
        raise ValueError("singular")

    broken.estimators = [("broken_estimator", "ATE", "ATE")]
    study = run_estimator_study(processed, [broken], scenarios=["A"], n_draws=1)
    assert study.estimates["estimate"].isna().all() and len(study.estimates) == 2 * 3
    assert "broken_error" in study.diagnostics
