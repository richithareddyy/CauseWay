"""Monte Carlo runner that applies estimators to every constructed sample."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from econml.dr import LinearDRLearner
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from causeway.balance import standardized_mean_difference
from causeway.benchmark import ENCODED_COLUMNS, Pair, build_pair, estimator_seed, oracle_ipw
from causeway.config import load_benchmark_config
from causeway.dml import doubleml_irm, hand_dml
from causeway.estimators import (
    ArmwiseRegression,
    cross_fit_folds,
    cross_fitted_scores,
    linear_propensity_model,
    outcome_regression,
)
from causeway.experiment import neyman_difference
from causeway.features import COVARIATE_COLUMNS
from causeway.matching import match_on_propensity, matched_att, matching_weights
from causeway.propensity import fit_logistic_propensity, overlap_metrics, predictive_metrics, recovery_metrics
from causeway.sensitivity import confounder_strength, ovb_analysis, rosenbaum_gamma, weighted_risk_ratio
from causeway.weighting import ate_weights, att_weights, hajek_difference, kish_ess, weight_diagnostics

CROSS_FIT_FOLDS = 5  # frozen v1: 5-fold cross-fitting for AIPW and DML
ROBUSTNESS_OUTCOMES = ["spend", "visit"]
PLACEBO_STREAM = 3100  # configs/robustness.toml


@dataclass
class ObservationalSample:
    comparison: str
    scenario: str
    draw: int
    outcomes: list[str]
    y: np.ndarray  # (n, k)
    treated: np.ndarray
    x_analyst: pd.DataFrame
    covariates: pd.DataFrame  # every encoded covariate, for balance diagnostics only
    row_ids: np.ndarray
    e_star: np.ndarray  # true design propensity: oracle and diagnostics only
    seed: int
    x_categorical: pd.DataFrame | None = None  # alt_categorical encoding, robustness checks only


def iter_samples(
    pairs: dict[str, Pair], spec: dict, scenarios: list[str], n_draws: int
) -> Iterator[ObservationalSample]:
    base_seed = spec["randomness"]["base_seed"]
    outcomes = [spec["outcomes"]["primary"], *spec["outcomes"]["secondary"]]
    for pair in pairs.values():
        y_all = pair.frame[outcomes].to_numpy(dtype=float)
        row_ids = pair.frame["row_id"].to_numpy()
        for scenario in scenarios:
            columns = pair.analyst_columns(scenario, spec)
            categorical = pair.analyst_categorical(scenario)
            for draw in range(n_draws):
                keep = pair.retained(scenario, draw, base_seed)
                yield ObservationalSample(
                    comparison=pair.name,
                    scenario=scenario,
                    draw=draw,
                    outcomes=outcomes,
                    y=y_all[keep],
                    treated=pair.treated[keep],
                    x_analyst=pair.covariates.loc[keep, columns].reset_index(drop=True),
                    covariates=pair.covariates.loc[keep].reset_index(drop=True),
                    row_ids=row_ids[keep],
                    e_star=pair.designs[scenario].e_star[keep],
                    seed=estimator_seed(base_seed, pair.index, draw),
                    x_categorical=categorical.loc[keep].reset_index(drop=True),
                )


Method = Callable[[ObservationalSample, dict], tuple[list[dict], dict]]


def produces(*estimators: tuple[str, str, str]) -> Callable[[Method], Method]:
    """Record the (estimator, estimand, benchmark) rows a method returns, so failures can be filled in."""

    def mark(method: Method) -> Method:
        method.estimators = list(estimators)
        return method

    return mark


def _rows(
    sample: ObservationalSample,
    estimator: str,
    estimand: str,
    benchmark: str,
    per_outcome: list[dict],
    **extra,
) -> list[dict]:
    base = {
        "comparison": sample.comparison,
        "scenario": sample.scenario,
        "draw": sample.draw,
        "estimator": estimator,
        "estimand": estimand,
        "benchmark": benchmark,
    }
    return [
        {**base, "outcome": outcome, **values, **extra}
        for outcome, values in zip(sample.outcomes, per_outcome, strict=True)
    ]


def _interval(result: dict) -> dict:
    return {key: result[key] for key in ("estimate", "se", "ci_low", "ci_high")}


def max_abs_smd(
    covariates: pd.DataFrame,
    treated: np.ndarray,
    weights: np.ndarray | None = None,
    columns: list[str] | None = None,
) -> float:
    """Largest |SMD| over the given covariates (default: all); weights adjust the means, not the scale."""
    w_t = None if weights is None else weights[treated]
    w_c = None if weights is None else weights[~treated]
    return float(
        max(
            abs(
                standardized_mean_difference(
                    covariates[c].to_numpy()[treated], covariates[c].to_numpy()[~treated], w_t, w_c
                )
            )
            for c in columns or COVARIATE_COLUMNS
        )
    )


def hidden_columns(sample: ObservationalSample, spec: dict) -> list[str]:
    """Encoded columns of the covariates hidden from the analyst in this sample's scenario."""
    return [
        c for raw in spec["scenarios"][sample.scenario]["hidden_covariates"] for c in ENCODED_COLUMNS[raw]
    ]


@produces(("naive_difference", "unadjusted association", "ATE"), ("oracle_ipw", "ATE", "ATE"))
def reference_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """Naive difference and the oracle IPW (true design propensity) as reference points."""
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    k = range(sample.y.shape[1])
    naive = [neyman_difference(sample.y[t, j], sample.y[~t, j], ci).__dict__ for j in k]
    oracle = [oracle_ipw(sample.y[:, j], t, sample.e_star, ci) for j in k]
    rows = _rows(sample, "naive_difference", "unadjusted association", "ATE", naive)
    rows += _rows(sample, "oracle_ipw", "ATE", "ATE", oracle)
    return rows, {"max_abs_smd_unadjusted": max_abs_smd(sample.covariates, t)}


@produces(("ipw_ate", "ATE", "ATE"), ("ipw_att", "ATT", "ATT"), ("matching_nn", "ATT", "ATT"))
def propensity_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """IPW (ATE and ATT) and matching, sharing one logistic propensity fit (frozen v1 rules)."""
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    k = range(sample.y.shape[1])
    fit = fit_logistic_propensity(sample.x_analyst, t)
    scores = fit.scores
    diagnostics = {
        "propensity_converged": fit.converged,
        **predictive_metrics(scores, t),
        **overlap_metrics(scores, t, spec["trimming"]["sensitivity_bounds"]),
        **recovery_metrics(scores, sample.e_star),
    }

    w_ate, w_att = ate_weights(scores, t), att_weights(scores, t)
    diagnostics.update(weight_diagnostics(w_ate, t, "ate"))
    diagnostics.update(weight_diagnostics(w_att, t, "att"))
    diagnostics["max_abs_smd_ipw_ate"] = max_abs_smd(sample.covariates, t, w_ate)
    # What an analyst can check: balance on the covariates the model was given (differs only in Scenario C).
    diagnostics["max_abs_smd_ipw_ate_visible"] = max_abs_smd(
        sample.covariates, t, w_ate, list(sample.x_analyst)
    )
    diagnostics["max_abs_smd_ipw_att"] = max_abs_smd(sample.covariates, t, w_att)
    ipw_ate = [hajek_difference(sample.y[:, j], t, w_ate, ci).__dict__ for j in k]
    ipw_att = [hajek_difference(sample.y[:, j], t, w_att, ci).__dict__ for j in k]

    caliper = next(e for e in spec["estimators"] if e["name"] == "matching_nn")["matching"][
        "caliper_sd_of_logit"
    ]
    match = match_on_propensity(scores, t, sample.row_ids, caliper)
    matched = [matched_att(sample.y[:, j], t, match, ci) for j in k]
    matched_weights = matching_weights(t, match)
    diagnostics.update(
        {
            "matching_caliper_logit": match.caliper,
            "matching_share_treated_discarded": matched[0]["share_treated_discarded"],
            "matching_n_pairs": matched[0]["n_pairs"],
            "matching_mean_distance": float(match.distances.mean()) if len(match.distances) else np.nan,
            "matching_max_distance": float(match.distances.max()) if len(match.distances) else np.nan,
            "max_abs_smd_matched": max_abs_smd(sample.covariates, t, matched_weights),
            "max_abs_smd_matched_visible": max_abs_smd(
                sample.covariates, t, matched_weights, list(sample.x_analyst)
            ),
        }
    )
    rows = _rows(
        sample,
        "ipw_ate",
        "ATE",
        "ATE",
        ipw_ate,
        ess_treated=diagnostics["ate_ess_treated"],
        ess_control=diagnostics["ate_ess_control"],
    )
    rows += _rows(
        sample,
        "ipw_att",
        "ATT",
        "ATT",
        ipw_att,
        ess_treated=diagnostics["att_ess_treated"],
        ess_control=diagnostics["att_ess_control"],
    )
    rows += _rows(
        sample,
        "matching_nn",
        "ATT",
        "ATT",
        [_interval(m) for m in matched],
        share_treated_discarded=matched[0]["share_treated_discarded"],
    )
    return rows, diagnostics


def _nuisance_metrics(
    prefix: str,
    y: np.ndarray,
    t: np.ndarray,
    propensity: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
    outcomes: list[str],
) -> dict:
    """Out-of-fold predictive quality of the nuisance models (predictive, not causal, evaluation)."""
    out = {f"{prefix}_{k}": v for k, v in predictive_metrics(propensity, t).items()}
    out[f"{prefix}_propensity_min"] = float(propensity.min())
    out[f"{prefix}_propensity_max"] = float(propensity.max())
    for j, outcome in enumerate(outcomes):
        residual = np.where(t, y[:, j] - mu1[:, j], y[:, j] - mu0[:, j])
        out[f"{prefix}_outcome_rmse_{outcome}"] = float(np.sqrt(np.mean(residual**2)))
    return out


@produces(("outcome_regression", "ATE", "ATE"), ("aipw", "ATE", "ATE"))
def regression_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """Outcome regression (arm-specific OLS) and cross-fitted AIPW."""
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    regression = [
        outcome_regression(sample.y[:, j], t, sample.x_analyst, ci).__dict__ for j in range(sample.y.shape[1])
    ]
    folds = cross_fit_folds(len(t), CROSS_FIT_FOLDS, sample.seed)
    fit = cross_fitted_scores(sample.y, t, sample.x_analyst, folds, linear_propensity_model, LinearRegression)
    rows = _rows(sample, "outcome_regression", "ATE", "ATE", regression)
    rows += _rows(sample, "aipw", "ATE", "ATE", [e.__dict__ for e in fit.estimates(ci)])
    return rows, _nuisance_metrics("aipw", sample.y, t, fit.propensity, fit.mu1, fit.mu0, sample.outcomes)


@produces(("dml_irm", "ATE", "ATE"))
def dml_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """DoubleML IRM with gradient-boosting nuisances and the draw's cross-fitting folds."""
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    folds = cross_fit_folds(len(t), CROSS_FIT_FOLDS, sample.seed)
    estimates, mu1, mu0, propensity = [], [], [], None
    for j in range(sample.y.shape[1]):
        estimate, predictions = doubleml_irm(sample.y[:, j], t, sample.x_analyst, folds, sample.seed, ci)
        estimates.append(estimate.__dict__)
        mu1.append(predictions["ml_g1"])
        mu0.append(predictions["ml_g0"])
        propensity = predictions["ml_m"] if propensity is None else propensity
    diagnostics = _nuisance_metrics(
        "dml", sample.y, t, propensity, np.column_stack(mu1), np.column_stack(mu0), sample.outcomes
    )
    return _rows(sample, "dml_irm", "ATE", "ATE", estimates), diagnostics


@produces(("aipw_econml", "ATE", "ATE"), ("dml_hand", "ATE", "ATE"))
def cross_check_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """EconML AIPW and the hand-written DML score, with the same models and folds as the primary versions."""
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    folds = cross_fit_folds(len(t), CROSS_FIT_FOLDS, sample.seed)
    econml_rows = []
    for j in range(sample.y.shape[1]):
        learner = LinearDRLearner(
            model_propensity=linear_propensity_model(),
            model_regression=ArmwiseRegression(),
            cv=folds,
            min_propensity=1e-12,
            random_state=sample.seed,
        )
        learner.fit(sample.y[:, j], t.astype(int), X=None, W=sample.x_analyst.to_numpy())
        low, high = learner.ate_interval(alpha=1 - ci)
        econml_rows.append(
            {"estimate": float(learner.ate()), "se": np.nan, "ci_low": float(low), "ci_high": float(high)}
        )
    hand = [e.__dict__ for e in hand_dml(sample.y, t, sample.x_analyst, folds, sample.seed).estimates(ci)]
    rows = _rows(sample, "aipw_econml", "ATE", "ATE", econml_rows)
    rows += _rows(sample, "dml_hand", "ATE", "ATE", hand)
    return rows, {}


def apply_methods(sample: ObservationalSample, methods: list[Method], spec: dict) -> tuple[list[dict], dict]:
    """Run every method on one sample. A method that raises is recorded as failed rows for that draw."""
    diagnostics = {
        "comparison": sample.comparison,
        "scenario": sample.scenario,
        "draw": sample.draw,
        "n": len(sample.treated),
    }
    estimate_rows = []
    for method in methods:
        try:
            rows, extra = method(sample, spec)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as error:
            failed = {"estimate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            rows = [
                row
                for name, estimand, benchmark in method.estimators
                for row in _rows(sample, name, estimand, benchmark, [failed] * len(sample.outcomes))
            ]
            extra = {f"{method.__name__}_error": repr(error)}
        estimate_rows += rows
        diagnostics.update(extra)
    return estimate_rows, diagnostics


def _apply_single_threaded(sample: ObservationalSample, methods: list[Method], spec: dict):
    with threadpool_limits(limits=1):
        return apply_methods(sample, methods, spec)


@dataclass
class EstimatorStudy:
    estimates: pd.DataFrame
    diagnostics: pd.DataFrame
    pairs: dict[str, Pair]


def run_estimator_study(
    processed: pd.DataFrame,
    methods: list[Method],
    spec: dict | None = None,
    scenarios: list[str] | None = None,
    n_draws: int | None = None,
    n_jobs: int = 1,
) -> EstimatorStudy:
    """Run the methods on every (comparison, scenario, draw). Results don't depend on n_jobs."""
    spec = spec or load_benchmark_config(1)
    scenarios = scenarios or list(spec["scenarios"])
    n_draws = n_draws or spec["randomness"]["n_draws"]
    pairs = {c["name"]: build_pair(processed, c, spec) for c in spec["comparisons"]}
    samples = iter_samples(pairs, spec, scenarios, n_draws)
    if n_jobs == 1:
        results = [apply_methods(sample, methods, spec) for sample in samples]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_apply_single_threaded)(s, methods, spec) for s in samples)
    estimates = pd.DataFrame([row for rows, _ in results for row in rows])
    diagnostics = pd.DataFrame([d for _, d in results])
    keys = ["comparison", "scenario", "draw"]
    if len(estimates):
        estimates = estimates.sort_values([*keys, "estimator", "outcome"], kind="stable").reset_index(
            drop=True
        )
    diagnostics = diagnostics.sort_values(keys, kind="stable").reset_index(drop=True)
    return EstimatorStudy(estimates, diagnostics, pairs)


# Robustness checks (configs/robustness.toml) run on spend and visit only.


def _restrict(sample: ObservationalSample, outcomes: list[str]) -> ObservationalSample:
    keep = [sample.outcomes.index(o) for o in outcomes]
    return replace(sample, outcomes=list(outcomes), y=sample.y[:, keep])


def _propensity_scores(sample: ObservationalSample, specification: str) -> np.ndarray:
    if specification == "main":
        return fit_logistic_propensity(sample.x_analyst, sample.treated).scores
    if specification == "alt_categorical":
        return fit_logistic_propensity(sample.x_categorical, sample.treated).scores
    x = sample.x_analyst.to_numpy()
    return (
        HistGradientBoostingClassifier(random_state=sample.seed).fit(x, sample.treated).predict_proba(x)[:, 1]
    )


@produces(
    ("ipw_ate", "ATE", "ATE"),
    ("ipw_att", "ATT", "ATT"),
    ("matching_nn", "ATT", "ATT"),
    ("aipw", "ATE", "ATE"),
)
def propensity_specification_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """IPW, matching and AIPW under each pre-registered propensity specification."""
    sample = _restrict(sample, ROBUSTNESS_OUTCOMES)
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    k = range(sample.y.shape[1])
    folds = cross_fit_folds(len(t), CROSS_FIT_FOLDS, sample.seed)
    propensity_models = {
        "main": (linear_propensity_model, None),
        "alt_categorical": (linear_propensity_model, sample.x_categorical),
        "alt_gbm": (lambda: HistGradientBoostingClassifier(random_state=sample.seed), None),
    }
    rows, diagnostics = [], {}
    for specification, (propensity_model, propensity_covariates) in propensity_models.items():
        scores = _propensity_scores(sample, specification)
        w_ate = ate_weights(scores, t)
        ipw_ate = [hajek_difference(sample.y[:, j], t, w_ate, ci).__dict__ for j in k]
        ipw_att = [hajek_difference(sample.y[:, j], t, att_weights(scores, t), ci).__dict__ for j in k]
        match = match_on_propensity(scores, t, sample.row_ids, 0.2)
        matched = [matched_att(sample.y[:, j], t, match, ci) for j in k]
        fit = cross_fitted_scores(
            sample.y,
            t,
            sample.x_analyst,
            folds,
            propensity_model,
            LinearRegression,
            propensity_covariates=propensity_covariates,
        )
        tag = {"specification": f"propensity={specification}"}
        rows += _rows(sample, "ipw_ate", "ATE", "ATE", ipw_ate, **tag)
        rows += _rows(sample, "ipw_att", "ATT", "ATT", ipw_att, **tag)
        rows += _rows(
            sample,
            "matching_nn",
            "ATT",
            "ATT",
            [_interval(m) for m in matched],
            share_treated_discarded=matched[0]["share_treated_discarded"],
            **tag,
        )
        rows += _rows(sample, "aipw", "ATE", "ATE", [e.__dict__ for e in fit.estimates(ci)], **tag)
        diagnostics[f"{specification}_ate_ess_share"] = kish_ess(w_ate) / len(t)
        diagnostics[f"{specification}_max_abs_smd_ipw"] = max_abs_smd(
            sample.covariates, t, w_ate, list(sample.x_analyst)
        )
    return rows, diagnostics


@produces(("outcome_regression", "ATE", "ATE"), ("aipw", "ATE", "ATE"))
def outcome_specification_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """Outcome regression and AIPW with gradient-boosting outcome models (alt_gbm)."""
    sample = _restrict(sample, ROBUSTNESS_OUTCOMES)
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    x = sample.x_analyst.to_numpy()
    folds = cross_fit_folds(len(t), CROSS_FIT_FOLDS, sample.seed)
    regression = []
    for j in range(sample.y.shape[1]):
        y = sample.y[:, j]
        mu1 = HistGradientBoostingRegressor(random_state=sample.seed).fit(x[t], y[t]).predict(x)
        mu0 = HistGradientBoostingRegressor(random_state=sample.seed).fit(x[~t], y[~t]).predict(x)
        # No analytic SE for g-computation with a boosted model; it is judged on bias and RMSE only.
        regression.append(
            {"estimate": float((mu1 - mu0).mean()), "se": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        )
    fit = cross_fitted_scores(
        sample.y,
        t,
        sample.x_analyst,
        folds,
        linear_propensity_model,
        lambda: HistGradientBoostingRegressor(random_state=sample.seed),
    )
    tag = {"specification": "outcome=alt_gbm"}
    rows = _rows(sample, "outcome_regression", "ATE", "ATE", regression, **tag)
    rows += _rows(sample, "aipw", "ATE", "ATE", [e.__dict__ for e in fit.estimates(ci)], **tag)
    return rows, {}


@produces(
    ("ipw_ate", "ATE_trimmed", "ATE_trimmed"),
    ("aipw", "ATE_trimmed", "ATE_trimmed"),
    ("dml_irm", "ATE_trimmed", "ATE_trimmed"),
)
def trimming_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """IPW, AIPW and DML restricted to units whose estimated propensity is inside each trimming band."""
    sample = _restrict(sample, ROBUSTNESS_OUTCOMES)
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    folds = cross_fit_folds(len(t), CROSS_FIT_FOLDS, sample.seed)
    scores = fit_logistic_propensity(sample.x_analyst, t).scores
    aipw = cross_fitted_scores(
        sample.y, t, sample.x_analyst, folds, linear_propensity_model, LinearRegression
    )
    dml = hand_dml(sample.y, t, sample.x_analyst, folds, sample.seed)
    rows, diagnostics = [], {}
    for low, high in spec["trimming"]["sensitivity_bounds"]:
        estimand = f"ATE_trimmed[{low:.2f},{high:.2f}]"
        tag = {"specification": f"trim=[{low:.2f},{high:.2f}]"}
        keep = (scores >= low) & (scores <= high)
        weights = np.where(keep, ate_weights(scores, t), 0.0)
        ipw = [hajek_difference(sample.y[:, j], t, weights, ci).__dict__ for j in range(sample.y.shape[1])]
        rows += _rows(sample, "ipw_ate", estimand, estimand, ipw, **tag)
        for name, fit in (("aipw", aipw), ("dml_irm", dml)):
            inside = (fit.propensity >= low) & (fit.propensity <= high)
            rows += _rows(
                sample, name, estimand, estimand, [e.__dict__ for e in fit.estimates(ci, inside)], **tag
            )
        diagnostics[f"share_kept_{low:.2f}"] = float(keep.mean())
    return rows, diagnostics


@produces(("matching_nn", "ATT", "ATT"))
def caliper_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """1:1 matching under each pre-registered caliper (main propensity model)."""
    sample = _restrict(sample, ROBUSTNESS_OUTCOMES)
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    scores = fit_logistic_propensity(sample.x_analyst, t).scores
    calipers = next(e for e in spec["estimators"] if e["name"] == "matching_nn")["matching"][
        "caliper_sensitivity"
    ]
    rows = []
    for caliper in calipers:
        match = match_on_propensity(scores, t, sample.row_ids, None if caliper == "none" else caliper)
        matched = [matched_att(sample.y[:, j], t, match, ci) for j in range(sample.y.shape[1])]
        rows += _rows(
            sample,
            "matching_nn",
            "ATT",
            "ATT",
            [_interval(m) for m in matched],
            share_treated_discarded=matched[0]["share_treated_discarded"],
            max_abs_smd=max_abs_smd(sample.covariates, t, matching_weights(t, match), list(sample.x_analyst)),
            specification=f"caliper={caliper}",
        )
    return rows, {}


@produces(("ipw_ate", "ATE", "ATE"), ("aipw", "ATE", "ATE"))
def refutation_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """Placebo treatment (everything refitted), an added random covariate, and an 80% subsample."""
    sample = _restrict(sample, ROBUSTNESS_OUTCOMES)
    ci = spec["estimation"]["ci_level"]
    rng = np.random.default_rng([PLACEBO_STREAM, sample.seed])

    def estimate(y, t, x):
        scores = fit_logistic_propensity(x, t).scores
        ipw = [hajek_difference(y[:, j], t, ate_weights(scores, t), ci).__dict__ for j in range(y.shape[1])]
        folds = cross_fit_folds(len(t), CROSS_FIT_FOLDS, sample.seed)
        aipw = cross_fitted_scores(y, t, x, folds, linear_propensity_model, LinearRegression)
        return {"ipw_ate": ipw, "aipw": [e.__dict__ for e in aipw.estimates(ci)]}

    y, t, x = sample.y, sample.treated, sample.x_analyst
    subset = rng.random(len(t)) < 0.8
    versions = {
        "original": estimate(y, t, x),
        "placebo": estimate(y, rng.permutation(t), x),
        "random_common_cause": estimate(y, t, x.assign(noise=rng.normal(size=len(t)))),
        "subset_80": estimate(y[subset], t[subset], x[subset].reset_index(drop=True)),
    }
    rows = []
    for version, results in versions.items():
        for name, values in results.items():
            rows += _rows(sample, name, "ATE", "ATE", values, specification=version)
    return rows, {}


@produces()
def sensitivity_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """Robustness values with covariate benchmarks, an E-value for visit and Rosenbaum Gamma.

    In Scenario C the hidden covariate's real strength is recorded too, which only a constructed
    benchmark allows.
    """
    t = sample.treated
    out = {}
    for outcome in ROBUSTNESS_OUTCOMES:
        y = sample.y[:, sample.outcomes.index(outcome)]
        summary, bounds = ovb_analysis(y, t, sample.x_analyst)
        out.update({f"{outcome}_{key}": value for key, value in summary.items()})
        for k, group in bounds.groupby("k"):
            strongest = group.loc[group["bias_bound"].idxmax()]
            out[f"{outcome}_max_bias_bound_k{int(k)}"] = strongest["bias_bound"]
            out[f"{outcome}_strongest_benchmark_k{int(k)}"] = strongest["benchmark"]
        for hidden in hidden_columns(sample, spec):
            strength = confounder_strength(y, t, sample.x_analyst, sample.covariates[hidden].to_numpy())
            out[f"{outcome}_hidden_{hidden}_r2_yz"] = strength["r2_yz"]
            out[f"{outcome}_hidden_{hidden}_r2_dz"] = strength["r2_dz"]
    scores = fit_logistic_propensity(sample.x_analyst, t).scores
    visit = sample.y[:, sample.outcomes.index("visit")]
    out.update(
        {
            f"visit_{key}": value
            for key, value in weighted_risk_ratio(visit, t, ate_weights(scores, t)).items()
        }
    )
    match = match_on_propensity(scores, t, sample.row_ids, 0.2)
    has_match = match.control_for_treated >= 0
    treated_visit = visit[t][has_match]
    control_visit = visit[~t][match.control_for_treated[has_match]]
    out["visit_matched_treated_only"] = int(((treated_visit == 1) & (control_visit == 0)).sum())
    out["visit_matched_control_only"] = int(((treated_visit == 0) & (control_visit == 1)).sum())
    out["visit_rosenbaum_gamma"] = rosenbaum_gamma(
        out["visit_matched_treated_only"], out["visit_matched_control_only"]
    )
    return [], out
