"""Monte Carlo runner that applies estimators to every constructed sample."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from causeway.balance import standardized_mean_difference
from causeway.benchmark import Pair, build_pair, estimator_seed, oracle_ipw
from causeway.config import load_benchmark_config
from causeway.experiment import neyman_difference
from causeway.features import COVARIATE_COLUMNS
from causeway.matching import match_on_propensity, matched_att, matching_weights
from causeway.propensity import fit_logistic_propensity, overlap_metrics, predictive_metrics, recovery_metrics
from causeway.weighting import ate_weights, att_weights, hajek_difference, weight_diagnostics


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
                )


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


def max_abs_smd(
    covariates: pd.DataFrame,
    treated: np.ndarray,
    weights: np.ndarray | None = None,
    columns: list[str] | None = None,
) -> float:
    """Largest |SMD| over the given covariates (default: all); weights adjust the means, not the scale."""
    values = []
    for column in columns or COVARIATE_COLUMNS:
        x = covariates[column].to_numpy()
        values.append(
            abs(
                standardized_mean_difference(
                    x[treated],
                    x[~treated],
                    None if weights is None else weights[treated],
                    None if weights is None else weights[~treated],
                )
            )
        )
    return float(max(values))


def reference_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """Naive difference and the oracle IPW (true design propensity) as reference points."""
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    naive = [
        neyman_difference(sample.y[t, k], sample.y[~t, k], ci).__dict__ for k in range(sample.y.shape[1])
    ]
    oracle = [oracle_ipw(sample.y[:, k], t, sample.e_star, ci) for k in range(sample.y.shape[1])]
    rows = _rows(sample, "naive_difference", "unadjusted association", "ATE", naive)
    rows += _rows(sample, "oracle_ipw", "ATE", "ATE", oracle)
    return rows, {"max_abs_smd_unadjusted": max_abs_smd(sample.covariates, t)}


def propensity_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """ipw_ate, ipw_att and matching_nn, all sharing one logistic propensity fit (frozen v1 rules)."""
    ci = spec["estimation"]["ci_level"]
    t = sample.treated
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
    ipw_ate = [hajek_difference(sample.y[:, k], t, w_ate, ci).__dict__ for k in range(sample.y.shape[1])]
    ipw_att = [hajek_difference(sample.y[:, k], t, w_att, ci).__dict__ for k in range(sample.y.shape[1])]

    matching_spec = next(e for e in spec["estimators"] if e["name"] == "matching_nn")["matching"]
    match = match_on_propensity(scores, t, sample.row_ids, matching_spec["caliper_sd_of_logit"])
    matched = [matched_att(sample.y[:, k], t, match, ci) for k in range(sample.y.shape[1])]
    diagnostics.update(
        {
            "matching_caliper_logit": match.caliper,
            "matching_share_treated_discarded": matched[0]["share_treated_discarded"],
            "matching_n_pairs": matched[0]["n_pairs"],
            "matching_mean_distance": float(match.distances.mean()) if len(match.distances) else np.nan,
            "matching_max_distance": float(match.distances.max()) if len(match.distances) else np.nan,
            "max_abs_smd_matched": max_abs_smd(sample.covariates, t, matching_weights(t, match)),
            "max_abs_smd_matched_visible": max_abs_smd(
                sample.covariates, t, matching_weights(t, match), list(sample.x_analyst)
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
        [{k: v for k, v in m.items() if k in ("estimate", "se", "ci_low", "ci_high")} for m in matched],
        share_treated_discarded=matched[0]["share_treated_discarded"],
    )
    return rows, diagnostics


reference_methods.estimators = [
    ("naive_difference", "unadjusted association", "ATE"),
    ("oracle_ipw", "ATE", "ATE"),
]
propensity_methods.estimators = [
    ("ipw_ate", "ATE", "ATE"),
    ("ipw_att", "ATT", "ATT"),
    ("matching_nn", "ATT", "ATT"),
]

Method = Callable[[ObservationalSample, dict], tuple[list[dict], dict]]


@dataclass
class EstimatorStudy:
    estimates: pd.DataFrame
    diagnostics: pd.DataFrame
    pairs: dict[str, Pair]


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


CROSS_FIT_FOLDS = 5  # frozen v1: 5-fold cross-fitting for AIPW and DML


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


def regression_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """outcome_regression (g-computation, arm-specific OLS) and aipw (hand-written, cross-fitted)."""
    from causeway.estimators import (
        cross_fit_folds,
        cross_fitted_scores,
        linear_outcome_model,
        linear_propensity_model,
        outcome_regression,
    )

    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    regression = [
        outcome_regression(sample.y[:, k], t, sample.x_analyst, ci).__dict__ for k in range(sample.y.shape[1])
    ]
    folds = cross_fit_folds(len(t), CROSS_FIT_FOLDS, sample.seed)
    fit = cross_fitted_scores(
        sample.y, t, sample.x_analyst, folds, linear_propensity_model, linear_outcome_model
    )
    aipw = [e.__dict__ for e in fit.estimates(ci)]
    rows = _rows(sample, "outcome_regression", "ATE", "ATE", regression)
    rows += _rows(sample, "aipw", "ATE", "ATE", aipw)
    return rows, _nuisance_metrics("aipw", sample.y, t, fit.propensity, fit.mu1, fit.mu0, sample.outcomes)


def dml_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """dml_irm: DoubleML IRM with gradient-boosting nuisances and the draw's cross-fitting folds."""
    from causeway.dml import doubleml_irm
    from causeway.estimators import cross_fit_folds

    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    folds = cross_fit_folds(len(t), CROSS_FIT_FOLDS, sample.seed)
    estimates, mu1, mu0, propensity = [], [], [], None
    for k in range(sample.y.shape[1]):
        estimate, predictions = doubleml_irm(sample.y[:, k], t, sample.x_analyst, folds, sample.seed, ci)
        estimates.append(estimate.__dict__)
        mu1.append(predictions["ml_g1"])
        mu0.append(predictions["ml_g0"])
        propensity = predictions["ml_m"] if propensity is None else propensity
    diagnostics = _nuisance_metrics(
        "dml", sample.y, t, propensity, np.column_stack(mu1), np.column_stack(mu0), sample.outcomes
    )
    return _rows(sample, "dml_irm", "ATE", "ATE", estimates), diagnostics


def cross_check_methods(sample: ObservationalSample, spec: dict) -> tuple[list[dict], dict]:
    """Library and hand-written counterparts of aipw and dml_irm, with identical nuisance models and folds:
    EconML's LinearDRLearner for AIPW, and the hand-written orthogonal score for DML."""
    from econml.dr import LinearDRLearner

    from causeway.dml import hand_dml
    from causeway.estimators import ArmwiseRegression, cross_fit_folds, linear_propensity_model

    ci = spec["estimation"]["ci_level"]
    t = sample.treated
    folds = cross_fit_folds(len(t), CROSS_FIT_FOLDS, sample.seed)
    econml_rows = []
    for k in range(sample.y.shape[1]):
        learner = LinearDRLearner(
            model_propensity=linear_propensity_model(),
            model_regression=ArmwiseRegression(),
            cv=folds,
            min_propensity=1e-12,
            random_state=sample.seed,
        )
        learner.fit(sample.y[:, k], t.astype(int), X=None, W=sample.x_analyst.to_numpy())
        low, high = learner.ate_interval(alpha=1 - ci)
        econml_rows.append(
            {"estimate": float(learner.ate()), "se": np.nan, "ci_low": float(low), "ci_high": float(high)}
        )
    hand = [e.__dict__ for e in hand_dml(sample.y, t, sample.x_analyst, folds, sample.seed).estimates(ci)]
    rows = _rows(sample, "aipw_econml", "ATE", "ATE", econml_rows)
    rows += _rows(sample, "dml_hand", "ATE", "ATE", hand)
    return rows, {}


regression_methods.estimators = [("outcome_regression", "ATE", "ATE"), ("aipw", "ATE", "ATE")]
dml_methods.estimators = [("dml_irm", "ATE", "ATE")]
cross_check_methods.estimators = [("aipw_econml", "ATE", "ATE"), ("dml_hand", "ATE", "ATE")]
