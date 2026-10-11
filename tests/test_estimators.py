"""Outcome regression, cross-fitted AIPW and DML: double robustness, cross-fitting and library agreement."""

import numpy as np
import pandas as pd
import pytest
from econml.dr import LinearDRLearner
from scipy import special
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from causeway.dml import doubleml_irm, hand_dml
from causeway.estimators import (
    ArmwiseRegression,
    cross_fit_folds,
    cross_fitted_scores,
    g_computation,
    linear_outcome_model,
    linear_propensity_model,
    outcome_regression,
)
from causeway.evaluation import dml_methods, reference_methods, regression_methods, run_estimator_study
from causeway.processing import TransformationLog, build_interim, build_processed
from conftest import make_raw_frame


def simulate(n: int, seed: int, strength: float = 1.0):
    """Confounded data: logistic propensity in x1, linear outcomes with an effect that varies in x2."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    e = special.expit(strength * x[:, 0])
    t = rng.random(n) < e
    y0 = 1.0 + 2.0 * x[:, 0] + x[:, 1] + rng.normal(size=n)
    y1 = y0 + 1.0 + 0.5 * x[:, 1]
    y = np.where(t, y1, y0)
    return pd.DataFrame(x, columns=["x1", "x2"]), t, y, float((y1 - y0).mean())


def constant_propensity():
    return DummyClassifier(strategy="prior")


def constant_outcome():
    return DummyRegressor(strategy="mean")


def test_g_computation_equals_the_reported_regression_estimate():
    x, t, y, _ = simulate(3000, 1)
    assert g_computation(y, t, x) == pytest.approx(outcome_regression(y, t, x).estimate, abs=1e-10)


def test_outcome_regression_recovers_ate_when_correctly_specified():
    x, t, y, ate = simulate(20000, 2)
    result = outcome_regression(y, t, x)
    assert abs(result.estimate - ate) < 4 * result.se


@pytest.mark.parametrize(
    ("propensity", "outcome", "consistent"),
    [
        (linear_propensity_model, linear_outcome_model, True),
        (linear_propensity_model, constant_outcome, True),  # outcome model wrong, propensity right
        (constant_propensity, linear_outcome_model, True),  # propensity wrong, outcome model right
        (constant_propensity, constant_outcome, False),  # both wrong: no protection
    ],
    ids=["both_right", "outcome_wrong", "propensity_wrong", "both_wrong"],
)
def test_aipw_is_doubly_robust(propensity, outcome, consistent):
    x, t, y, ate = simulate(40000, 3)
    fit = cross_fitted_scores(y, t, x, cross_fit_folds(len(t), 5, 0), propensity, outcome)
    estimate = fit.estimates()[0]
    if consistent:
        assert abs(estimate.estimate - ate) < 4 * estimate.se
    else:
        assert abs(estimate.estimate - ate) > 10 * estimate.se


def test_aipw_score_formula_with_fixed_nuisances():
    class Half(DummyClassifier):
        def predict_proba(self, x):
            return np.full((len(x), 2), 0.5)

    y = np.array([4.0, 2.0, 1.0, 3.0, 5.0, 0.0])
    t = np.array([True, False, True, False, True, False])
    x = pd.DataFrame({"x": np.arange(6.0)})
    fit = cross_fitted_scores(
        y,
        t,
        x,
        cross_fit_folds(6, 3, 0),
        lambda: Half(),
        lambda: DummyRegressor(strategy="constant", constant=0.0),
    )
    np.testing.assert_allclose(fit.psi[:, 0], np.where(t, 2 * y, -2 * y))


def test_cross_fitting_never_uses_a_units_own_outcome():
    rng = np.random.default_rng(4)
    x = pd.DataFrame({"x": rng.normal(size=500)})
    t = rng.random(500) < 0.5
    y = rng.normal(size=500)
    folds = cross_fit_folds(500, 5, 0)
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(500))
    fit = cross_fitted_scores(
        y, t, x, folds, linear_propensity_model, lambda: KNeighborsRegressor(n_neighbors=1)
    )
    own = np.where(t, fit.mu1[:, 0], fit.mu0[:, 0])
    assert not np.isclose(own, y).any()  # a 1-NN learner would reproduce y in-sample


def test_aipw_matches_econml_with_identical_models_and_folds():
    x, t, y, _ = simulate(3000, 5)
    folds = cross_fit_folds(len(t), 5, 7)
    hand = cross_fitted_scores(y, t, x, folds, linear_propensity_model, linear_outcome_model).estimates()[0]
    learner = LinearDRLearner(
        model_propensity=linear_propensity_model(),
        model_regression=ArmwiseRegression(),
        cv=folds,
        min_propensity=1e-12,
        random_state=0,
    )
    learner.fit(y, t.astype(int), X=None, W=x.to_numpy())
    assert hand.estimate == pytest.approx(float(learner.ate()), abs=1e-10)


def test_dml_hand_matches_doubleml_including_extreme_propensities():
    x, t, y, _ = simulate(3000, 6, strength=4.0)  # strong selection: fitted propensities fall below 0.01
    folds = cross_fit_folds(len(t), 5, 11)
    hand = hand_dml(y, t, x, folds, seed=11)
    library, predictions = doubleml_irm(y, t, x, folds, seed=11)
    assert predictions["ml_m"].min() < 0.01  # DoubleML's default clipping would have changed the answer
    assert library.estimate == pytest.approx(hand.estimates()[0].estimate, abs=1e-10)
    np.testing.assert_allclose(predictions["ml_m"], hand.propensity)
    # DoubleML uses the population variance of the score; the hand version uses n - 1.
    assert library.se == pytest.approx(hand.estimates()[0].se * np.sqrt((len(t) - 1) / len(t)), rel=1e-9)


def test_armwise_regression_fits_each_arm_separately():
    x = np.array([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
    t = np.array([1, 1, 1, 0, 0, 0])
    y = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])
    model = ArmwiseRegression(LinearRegression()).fit(np.column_stack([x, t]), y)
    np.testing.assert_allclose(model.predict(np.column_stack([x, t])), y, atol=1e-12)


@pytest.fixture(scope="module")
def processed():
    log = TransformationLog()
    return build_processed(build_interim(make_raw_frame(n_per_arm=1500, seed=9), log), log)


def test_parallel_and_sequential_studies_are_identical(processed):
    methods = [reference_methods, regression_methods, dml_methods]
    sequential = run_estimator_study(processed, methods, scenarios=["A"], n_draws=2, n_jobs=1)
    parallel = run_estimator_study(processed, methods, scenarios=["A"], n_draws=2, n_jobs=2)
    pd.testing.assert_frame_equal(sequential.estimates, parallel.estimates)
    assert set(sequential.estimates["estimator"]) == {
        "naive_difference",
        "oracle_ipw",
        "outcome_regression",
        "aipw",
        "dml_irm",
    }
    assert (
        sequential.estimates.loc[sequential.estimates["estimator"].isin(["aipw", "dml_irm"]), "estimand"]
        == "ATE"
    ).all()
