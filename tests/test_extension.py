"""Validation extension v1 on a synthetic randomized pool (the Criteo file is not needed)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from causeway import extension as ex
from causeway.weighting import ate_weights, att_weights, hajek_difference

FEATURES = [f"f{i}" for i in range(12)]


def synthetic_pool(n: int = 20000, seed: int = 0, effect: float = 0.03) -> pd.DataFrame:
    """Randomized at 85%, with visit depending on the features and a constant effect."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 12)), columns=FEATURES)
    x["f0"] = rng.choice([12.6, 20.0, 24.0, 26.0], size=n) + rng.normal(0, 0.5, n)
    t = rng.random(n) < 0.85
    base = expit(-3.0 + 0.4 * x["f6"] + 0.3 * x["f9"] - 0.2 * x["f2"])
    visit = rng.random(n) < np.clip(base + effect * t, 0, 1)
    conversion = visit & (rng.random(n) < 0.1)
    return x.assign(treatment=t, visit=visit.astype(float), conversion=conversion.astype(float))


@pytest.fixture(scope="module")
def protocol():
    plan = ex.load_protocol()
    plan["cross_checks"]["draws"] = 1
    return plan


@pytest.fixture(scope="module")
def pool():
    return synthetic_pool()


def test_intercept_matches_the_target_mean_propensity(pool, protocol):
    designs = ex.make_designs(pool, protocol)
    share = pool["treatment"].mean()
    for design in designs.values():
        assert design.e_star.mean() == pytest.approx(share, abs=1e-9)
        assert design.keep_probability.max() == pytest.approx(1.0)
        assert (design.keep_probability >= 0).all()
    assert designs["H"].hidden == ["f6"]
    np.testing.assert_array_equal(designs["A"].e_star, designs["H"].e_star)  # same selection, less seen
    assert designs["P"].e_star.std() > designs["A"].e_star.std()


def test_scenarios_share_each_draws_selection_uniforms():
    a, seed_a = ex.draw_seeds(1, 7)
    b, seed_b = ex.draw_seeds(1, 7)
    np.testing.assert_array_equal(a.random(5), b.random(5))
    assert seed_a == seed_b and ex.draw_seeds(1, 8)[1] != seed_a


def test_references_have_their_own_definitions(pool, protocol):
    designs = ex.make_designs(pool, protocol)
    refs = ex.references(pool, designs, protocol)
    t, y = pool["treatment"].to_numpy(), pool["visit"].to_numpy()
    a = refs[(refs["scenario"] == "A") & (refs["outcome"] == "visit")].set_index("estimand")
    assert a.loc["ATE", "value"] == pytest.approx(y[t].mean() - y[~t].mean())
    assert a.loc["ATT", "value"] == pytest.approx(hajek_difference(y, t, designs["A"].e_star).estimate)
    inside = (designs["A"].e_star >= 0.05) & (designs["A"].e_star <= 0.95)
    assert a.loc["ATE_trimmed", "share_of_pool"] == pytest.approx(inside.mean())
    assert set(refs["estimand"]) == {"ATE", "ATE_adjusted", "ATT", "ATE_trimmed"}
    assert (refs["se"] > 0).all()


def test_oracles_are_unbiased_for_their_references(pool, protocol):
    designs = ex.make_designs(pool, protocol)
    t, y = pool["treatment"].to_numpy(), pool["visit"].to_numpy()
    for name in ("A", "N", "P"):
        e = designs[name].e_star
        refs = {"ATE": y[t].mean() - y[~t].mean(), "ATT": hajek_difference(y, t, e).estimate}
        errors = {"ATE": [], "ATT": []}
        for draw in range(300):
            keep = ex.draw_seeds(5, draw)[0].random(len(t)) <= designs[name].keep_probability
            tk, yk, ek = t[keep], y[keep], e[keep]
            errors["ATE"].append(hajek_difference(yk, tk, ate_weights(ek, tk)).estimate - refs["ATE"])
            errors["ATT"].append(hajek_difference(yk, tk, att_weights(ek, tk)).estimate - refs["ATT"])
        for estimand, values in errors.items():
            values = np.asarray(values)
            assert abs(values.mean()) < 3 * values.std(ddof=1) / np.sqrt(len(values)), (name, estimand)


def test_nu2_warnings_are_counted_and_still_emitted():
    def noisy():
        warnings.warn(
            "The estimated nu2 for _d is not positive. Re-estimation ...", UserWarning, stacklevel=2
        )
        warnings.warn("something else", RuntimeWarning, stacklevel=2)
        return 1

    with pytest.warns(UserWarning, match="nu2") as record:
        result, count = ex._counting_nu2(noisy)
    assert result == 1 and count == 1
    assert {w.category for w in record} == {UserWarning, RuntimeWarning}


def test_a_failed_sample_counts_against_every_estimator(pool, protocol, monkeypatch):
    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(ex, "analyse_sample", broken)
    refs, estimates, diagnostics = ex.run_study(pool, protocol, n_draws=1, n_jobs=1)
    assert len(estimates) == len(protocol["estimators"]) * 2 * len(protocol["scenarios"])
    assert estimates["estimate"].isna().all() and diagnostics["error"].str.contains("singular").all()
    summary, _ = ex.summarize(refs, estimates)
    assert (summary["failure_rate"] == 1.0).all()


def test_end_to_end_with_cross_checks(protocol):
    small = synthetic_pool(n=8000, seed=3)
    refs, estimates, diagnostics = ex.run_study(small, protocol, n_draws=2, n_jobs=1)
    expected = {e["name"] for e in protocol["estimators"]} | set(ex.CROSS_CHECKS)
    assert set(estimates["estimator"]) == expected
    assert set(estimates.loc[estimates["estimator"] == "dml_irm_att", "benchmark"]) == {"ATT"}
    assert set(estimates.loc[estimates["estimator"] == "naive_difference", "estimand"]) == {
        "unadjusted association"
    }
    assert diagnostics["dml_fits"].eq(4).all() and diagnostics["nu2_fallbacks"].ge(0).all()
    assert "error" not in diagnostics or diagnostics["error"].isna().all()  # no whole sample is lost
    # a refused DoubleML fit fails only its own rows
    failed = estimates[estimates["estimator"].str.startswith("dml_irm") & estimates["estimate"].isna()]
    assert len(failed) == diagnostics["dml_failed_fits"].sum()
    others = estimates[~estimates["estimator"].str.startswith("dml_irm")]
    assert others["estimate"].notna().all()
    checks = ex.cross_check_table(estimates)
    assert set(checks["check"]) == {"aipw_econml vs aipw", "dml_hand vs dml_irm"}
    assert (checks["max_abs_difference"] < 1e-8).all()
    summary, versus_adjusted = ex.summarize(refs, estimates)
    assert not summary["estimator"].isin(ex.CROSS_CHECKS).any()
    assert set(versus_adjusted["benchmark"]) == {"ATE"}
    assert summary["oracle_coverage"].notna().all()


def test_report_renders_from_a_small_run(protocol, tmp_path):
    from causeway import visualization as viz

    small = synthetic_pool(n=12000, seed=4)
    refs, estimates, diagnostics = ex.run_study(small, protocol, n_draws=2, n_jobs=1)
    summary, versus_adjusted = ex.summarize(refs, estimates)
    tables = {
        "references": refs,
        "design": ex.design_summary(small, protocol),
        "estimates": estimates,
        "diagnostics": diagnostics,
        "summary": summary,
        "summary_vs_adjusted": versus_adjusted,
        "cross_checks": ex.cross_check_table(estimates),
    }
    provenance = {
        "data": {
            "pool_rows": 12000,
            "pool_seed": 1,
            "raw_rows": 12000,
            "raw_sha256": "a" * 64,
            "treatment_share": 0.85,
            "pool_sha256": "b" * 64,
        },
        "protocol": {"sha256": "c" * 64},
        "gate": {"sha256": "d" * 64},
        "code": {"commit": "abcdef0", "uncommitted_changes": False},
        "environment": {"python": "3.12", "packages": {"numpy": "2"}},
    }
    tables["post_hoc_reference_checks"] = ex.reference_checks(small, protocol)
    tables["post_hoc_hidden_shift"] = ex.hidden_shift(estimates, refs)
    text = ex.extension_markdown(tables, protocol, provenance)
    assert (
        "## 7. Post-hoc analyses (not in the frozen protocol)" in text and "## 8. Reading the results" in text
    )
    for heading in ("## 1. Data and references", "## 3. Estimator recovery", "## 9. Expectations"):
        assert heading in text
    assert "true answer" not in text and "the truth" not in text
    figure = viz.extension_heatmap(summary, tmp_path / "h.png", ex.FIGURE_ESTIMATORS, ["visit", "conversion"])
    assert figure.exists() and figure.stat().st_size > 0


def test_hidden_shift_is_paired_by_draw():
    rows = []
    for draw in range(4):
        for scenario, value in (("A", 1.0 + draw), ("H", 1.5 + draw)):
            rows.append(
                {
                    "scenario": scenario,
                    "draw": draw,
                    "estimator": "aipw",
                    "outcome": "visit",
                    "estimate": value,
                }
            )
    refs = pd.DataFrame({"scenario": ["A"], "estimand": ["ATE"], "outcome": ["visit"], "value": [2.0]})
    shift = ex.hidden_shift(pd.DataFrame(rows), refs).iloc[0]
    # the draw-to-draw variation cancels: every paired difference is exactly 0.5
    assert shift["mean_shift"] == pytest.approx(0.5) and shift["mc_se"] == pytest.approx(0.0)
    assert shift["shift_pct_of_difference_in_means"] == pytest.approx(0.25) and shift["draws"] == 4
