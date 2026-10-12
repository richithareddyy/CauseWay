"""Heterogeneity tests and targeting on simulated randomized experiments with known effects."""

import copy

import numpy as np
import pandas as pd
import pytest

from causeway.heterogeneity import (
    Z_POWER,
    aggregate_splits,
    blp_gates_clan,
    classify,
    equality_test,
    heterogeneity_verdict,
    load_heterogeneity_plan,
    run_split,
    segment_effects,
)
from causeway.targeting import (
    CrossFittedEffects,
    evaluate_policies,
    policies,
    policy_value,
    top_fraction,
    uplift_curve,
)


def experiment(n: int, seed: int, heterogeneity: float):
    """Randomized experiment with tau(x) = 1 + heterogeneity * x."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    t = rng.random(n) < 0.5
    tau = 1.0 + heterogeneity * x[:, 0]
    y = x[:, 1] + t * tau + rng.normal(size=n)
    return y, t, x, tau


def run_blp(y, t, x, proxy):
    return blp_gates_clan(y, t, x, ["x0", "x1"], proxy, x[:, 1], n_groups=5, ci_level=0.95)


def test_blp_recovers_a_perfect_proxy():
    y, t, x, tau = experiment(40_000, 1, heterogeneity=1.0)
    out = run_blp(y, t, x, tau)
    assert out["b1"]["estimate"] == pytest.approx(tau.mean(), abs=0.05)
    assert out["b2"]["estimate"] == pytest.approx(1.0, abs=0.1)
    assert out["gate_top_minus_bottom"]["ci_low"] > 0
    assert out["clan_x0"]["estimate"] > 2.0  # the most-affected group has much larger x0


def test_blp_does_not_reward_an_uninformative_proxy():
    y, t, x, _ = experiment(40_000, 2, heterogeneity=1.0)
    out = run_blp(y, t, x, np.random.default_rng(3).normal(size=len(y)))
    assert out["b2"]["ci_low"] < 0 < out["b2"]["ci_high"]


def test_gates_are_flat_without_heterogeneity():
    y, t, x, _ = experiment(40_000, 4, heterogeneity=0.0)
    out = run_blp(y, t, x, x[:, 0])  # a proxy that varies, but effects do not
    assert out["gate_top_minus_bottom"]["ci_low"] < 0 < out["gate_top_minus_bottom"]["ci_high"]


def test_aggregation_uses_medians_and_doubles_the_median_p_value():
    splits = [
        {"b2": {"estimate": e, "ci_low": e - 1, "ci_high": e + 1, "p_value": p, "se": 0.5}}
        for e, p in ((1.0, 0.01), (2.0, 0.02), (3.0, 0.40))
    ]
    row = aggregate_splits(splits).iloc[0]
    assert (row["estimate"], row["ci_low"], row["ci_high"]) == (2.0, 1.0, 3.0)
    assert row["p_value"] == pytest.approx(0.04)
    capped = aggregate_splits([{"b2": {"estimate": 0, "ci_low": -1, "ci_high": 1, "p_value": 0.8, "se": 1}}])
    assert capped.iloc[0]["p_value"] == 1.0


def verdict_table(b2, b2_p, spread_low, spread_high):
    return pd.DataFrame(
        [
            {"parameter": "b2", "estimate": b2, "ci_low": 0, "ci_high": 0, "p_value": b2_p},
            {
                "parameter": "gate_top_minus_bottom",
                "estimate": 0.0,
                "ci_low": spread_low,
                "ci_high": spread_high,
                "p_value": 0.5,
            },
        ]
    )


def test_verdict_requires_both_blp_and_gates_evidence():
    assert heterogeneity_verdict(verdict_table(0.8, 0.01, 0.1, 0.5), 0.05)["heterogeneity_detected"]
    assert not heterogeneity_verdict(verdict_table(0.8, 0.01, -0.1, 0.5), 0.05)["heterogeneity_detected"]
    assert not heterogeneity_verdict(verdict_table(0.8, 0.20, 0.1, 0.5), 0.05)["heterogeneity_detected"]
    assert not heterogeneity_verdict(verdict_table(-0.8, 0.01, 0.1, 0.5), 0.05)["heterogeneity_detected"]


def test_run_split_end_to_end_detects_strong_heterogeneity():
    plan = copy.deepcopy(load_heterogeneity_plan())
    plan["causal_forest"] = {**plan["causal_forest"], "n_estimators": 100, "min_samples_leaf": 20}
    y, t, x, _ = experiment(6000, 5, heterogeneity=2.0)
    out = run_split(y, t, x, ["x0", "x1"], plan, seed=1)
    assert out["b2"]["estimate"] > 0.5 and out["b2"]["p_value"] < 0.01


def test_segment_effects_and_minimum_detectable_effect():
    y = np.array([3.0, 5.0, 1.0, 1.0, 2.0, 4.0, 0.0, 2.0])
    t = np.array([True, True, False, False, True, True, False, False])
    levels = pd.Series(["a", "a", "a", "a", "b", "b", "b", "b"])
    table = segment_effects(y, t, levels, "group").set_index("level")
    assert table.loc["a", "estimate"] == pytest.approx(3.0)
    assert table.loc["b", "estimate"] == pytest.approx(2.0)
    assert table.loc["a", "mde"] == pytest.approx(Z_POWER * table.loc["a", "se"])


def test_equality_test():
    same = pd.DataFrame({"estimate": [1.0, 1.0, 1.0], "se": [0.1, 0.2, 0.3]})
    assert equality_test(same)["chi2"] == pytest.approx(0.0)
    different = pd.DataFrame({"estimate": [0.0, 1.0], "se": [0.1, 0.1]})
    result = equality_test(different)
    assert result["chi2"] == pytest.approx(50.0) and result["dof"] == 1 and result["p_value"] < 1e-10


def test_classification_rule():
    assert classify(0.5, 0.01) == "benefit"
    assert classify(-0.5, 0.01) == "harm"
    assert classify(0.5, 0.2) == "inconclusive"


@pytest.fixture
def three_arm():
    """Mens helps customers with x > 0, Womens helps those with x < 0, nobody gains from both."""
    rng = np.random.default_rng(7)
    n = 60_000
    x = rng.normal(size=n)
    arm = rng.integers(0, 3, size=n)
    effect = {1: np.where(x > 0, 2.0, -1.0), 2: np.where(x < 0, 2.0, -1.0)}
    y = rng.normal(size=n) + np.select([arm == 1, arm == 2], [effect[1], effect[2]], 0.0)
    return CrossFittedEffects(arm, y, {"mens": effect[1], "womens": effect[2]})


def test_treat_all_policy_value_is_the_arm_mean(three_arm):
    value = policy_value(np.ones(len(three_arm.y), dtype=int), three_arm.arm, three_arm.y)
    assert value == pytest.approx(three_arm.y[three_arm.arm == 1].mean())


def test_top_fraction_selects_the_highest_scores():
    selected = top_fraction(np.array([0.1, 0.9, 0.5, 0.7]), 0.5)
    np.testing.assert_array_equal(selected, [False, True, False, True])


def test_best_arm_policy_beats_treating_everyone_when_heterogeneity_is_real(three_arm):
    values, differences = evaluate_policies(
        three_arm, budget=0.2, n_bootstrap=200, seed=1, comparisons=[("model_best_arm", "treat_all_mens")]
    )
    table = values.set_index("policy")
    assert table.loc["model_best_arm", "incremental_value"] == pytest.approx(2.0, abs=0.1)
    assert table.loc["treat_all_mens", "incremental_value"] == pytest.approx(0.5, abs=0.1)
    assert table.loc["treat_none", "incremental_value"] == 0.0
    assert differences.iloc[0]["interval_excludes_zero"]
    best = policies(three_arm, 0.2, np.random.default_rng(0))["model_best_arm"]
    assert set(np.unique(best)) == {1, 2}


def test_uplift_curve_endpoints(three_arm):
    curve = uplift_curve(three_arm, "mens", np.array([0.0, 1.0])).set_index("fraction_targeted")
    assert curve.loc[0.0, "model_incremental_value"] == pytest.approx(0.0)
    assert curve.loc[1.0, "model_incremental_value"] == pytest.approx(
        curve.loc[1.0, "random_incremental_value"]
    )


def test_sign_unstable_parameters_are_flagged_by_share_positive():
    # Each split "rejects", but in opposite directions: the median p-value is small while the estimate is ~0.
    splits = [
        {"clan_x": {"estimate": e, "ci_low": e - 0.01, "ci_high": e + 0.01, "p_value": 1e-6, "se": 0.005}}
        for e in (0.05, -0.05, 0.05, -0.05, 0.0)
    ]
    row = aggregate_splits(splits).iloc[0]
    assert row["p_value"] < 1e-5 and row["estimate"] == 0.0
    assert row["share_positive"] == pytest.approx(0.4)
