"""Part II: the constructed observational benchmark (configs/benchmark/v1.toml).

Randomization is removed by RCT rejection sampling (Keith et al., 2023): within each randomized pair,
unit i with trial arm t_i is kept with probability P*(T=t_i | x_i) / (P_hat(T=t_i) * M). In the retained
sample P(T=1 | X) equals the design propensity e*(X) and the covariate distribution is unchanged in
expectation, so the confounding is real but entirely explained by observed covariates.

Nothing here is Hillstrom's own data-generating process. Every retained sample is reproducible from the
frozen base seed, the comparison index and the draw index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import special, stats

from causeway.balance import standardized_mean_difference
from causeway.config import load_benchmark_config
from causeway.experiment import neyman_difference
from causeway.features import COVARIATE_COLUMNS, covariate_matrix

# Encoded columns derived from each raw covariate named in the specification.
ENCODED_COLUMNS = {
    "history": ["ln_history"],
    "recency": ["recency"],
    "mens": ["mens"],
    "womens": ["womens"],
    "newbie": ["newbie"],
    "zip_code": ["zip_urban", "zip_rural"],
    "channel": ["channel_web", "channel_multichannel"],
}
SELECTION_STREAM, ESTIMATOR_STREAM = 0, 1


class OverlapGuardError(RuntimeError):
    """The pre-specified covariate-only overlap guard failed; a recorded amendment is required."""


# --------------------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------------------


def seed_sequence(
    base_seed: int, comparison_index: int, draw_index: int, stream: int
) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy=base_seed, spawn_key=(comparison_index, draw_index, stream))


def selection_uniforms(base_seed: int, comparison_index: int, draw_index: int, n: int) -> np.ndarray:
    """Uniforms for one draw, one per pair unit in row_id order; shared by every scenario."""
    rng = np.random.default_rng(seed_sequence(base_seed, comparison_index, draw_index, SELECTION_STREAM))
    return rng.random(n)


def estimator_seed(base_seed: int, comparison_index: int, draw_index: int) -> int:
    """Integer seed for any randomness inside estimators (folds, learners) for one draw."""
    return int(seed_sequence(base_seed, comparison_index, draw_index, ESTIMATOR_STREAM).generate_state(1)[0])


# --------------------------------------------------------------------------------------------------
# Design propensity and selection
# --------------------------------------------------------------------------------------------------


def selection_terms(frame: pd.DataFrame) -> pd.DataFrame:
    """The four terms of the frozen selection model, z-scored on the pair population (ddof = 0)."""

    def z(values: pd.Series) -> pd.Series:
        return (values - values.mean()) / values.std(ddof=0)

    return pd.DataFrame(
        {
            "z_log_history": z(np.log(frame["history"].astype(float))),
            "z_recency": z(frame["recency"].astype(float)),
            "multichannel": (frame["channel"].astype(str) == "Multichannel").astype(float),
            "newbie": frame["newbie"].astype(float),
        },
        index=frame.index,
    )


def linear_index(terms: pd.DataFrame, spec: dict) -> np.ndarray:
    coefficients = {term["name"]: term["coefficient"] for term in spec["selection"]["terms"]}
    return terms[list(coefficients)].to_numpy() @ np.array(list(coefficients.values()))


def calibrate_intercept(index: np.ndarray, k: float, target: float, tolerance: float) -> float:
    """Solve mean(expit(alpha + k * index)) = target for alpha by bisection."""
    low, high = -50.0, 50.0
    while high - low > tolerance:
        mid = (low + high) / 2
        if special.expit(mid + k * index).mean() < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def acceptance_probabilities(e_star: np.ndarray, treated: np.ndarray) -> tuple[np.ndarray, float]:
    """Rejection-sampling acceptance probabilities and the bound M (Keith et al., 2023, Algorithm 1)."""
    share_treated = treated.mean()
    ratio = np.where(treated, e_star / share_treated, (1 - e_star) / (1 - share_treated))
    m = float(ratio.max())
    return ratio / m, m


@dataclass
class ScenarioDesign:
    name: str
    k: float
    hidden: list[str]
    alpha: float
    e_star: np.ndarray
    acceptance: np.ndarray
    m: float


@dataclass
class Pair:
    """One randomized pair (an e-mail arm and the control arm) with every scenario's selection design."""

    name: str
    index: int
    frame: pd.DataFrame
    treated: np.ndarray
    covariates: pd.DataFrame
    designs: dict[str, ScenarioDesign] = field(default_factory=dict)

    def retained(self, scenario: str, draw: int, base_seed: int) -> np.ndarray:
        uniforms = selection_uniforms(base_seed, self.index, draw, len(self.frame))
        return uniforms <= self.designs[scenario].acceptance

    def analyst_columns(self, scenario: str, spec: dict) -> list[str]:
        hidden = set(self.designs[scenario].hidden)
        return [c for raw in spec["analyst"]["covariates"] if raw not in hidden for c in ENCODED_COLUMNS[raw]]


def build_pair(processed: pd.DataFrame, comparison: dict, spec: dict) -> Pair:
    arms = [comparison["treatment_arm"], comparison["control_arm"]]
    frame = processed[processed["segment"].isin(arms)].sort_values("row_id").reset_index(drop=True)
    treated = (frame["segment"] == comparison["treatment_arm"]).to_numpy()
    pair = Pair(comparison["name"], comparison["index"], frame, treated, covariate_matrix(frame))
    index = linear_index(selection_terms(frame), spec)
    selection = spec["selection"]
    for name, scenario in spec["scenarios"].items():
        k = scenario["selection_strength_k"]
        alpha = calibrate_intercept(
            index, k, selection["intercept_target_mean_propensity"], selection["intercept_tolerance"]
        )
        e_star = special.expit(alpha + k * index)
        acceptance, m = acceptance_probabilities(e_star, treated)
        pair.designs[name] = ScenarioDesign(
            name, k, list(scenario["hidden_covariates"]), alpha, e_star, acceptance, m
        )
    return pair


def overlap_guard(pair: Pair, spec: dict) -> list[dict]:
    """Covariate-only check of the design propensity, run before any outcome is used."""
    guard = spec["selection"]["overlap_guard"]
    low, high = guard["extreme_bounds"]
    rows = []
    for name, design in pair.designs.items():
        share_outside = float(((design.e_star < low) | (design.e_star > high)).mean())
        applies = name in guard["applies_to"]
        rows.append(
            {
                "comparison": pair.name,
                "scenario": name,
                "k": design.k,
                "alpha": design.alpha,
                "e_star_min": float(design.e_star.min()),
                "e_star_max": float(design.e_star.max()),
                "share_outside_bounds": share_outside,
                "guard_applies": applies,
                "passed": (share_outside <= guard["max_share_outside_bounds"]) if applies else None,
            }
        )
    failed = [r for r in rows if r["passed"] is False]
    if failed:
        raise OverlapGuardError(
            f"overlap guard failed for {failed}; record a v1.1 amendment before proceeding"
        )
    return rows


# --------------------------------------------------------------------------------------------------
# Experimental benchmarks (full randomized pair)
# --------------------------------------------------------------------------------------------------


def _hajek_difference(y: np.ndarray, treated: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted difference in Hajek means with a linearization standard error."""
    out, var = [], 0.0
    for mask in (treated, ~treated):
        w, values = weights[mask], y[mask]
        mean = (w * values).sum() / w.sum()
        var += ((w * (values - mean)) ** 2).sum() / w.sum() ** 2
        out.append(mean)
    return float(out[0] - out[1]), float(np.sqrt(var))


def experimental_benchmarks(pair: Pair, spec: dict) -> pd.DataFrame:
    """Finite-population targets computed on the full randomized pair, matched to each estimand."""
    outcomes = [spec["outcomes"]["primary"], *spec["outcomes"]["secondary"]]
    rows = []
    for name, design in pair.designs.items():
        for outcome in outcomes:
            y = pair.frame[outcome].to_numpy(dtype=float)
            base = {"comparison": pair.name, "scenario": name, "outcome": outcome}
            ate = neyman_difference(y[pair.treated], y[~pair.treated])
            rows.append({**base, "estimand": "ATE", "value": ate.estimate, "se": ate.se, "n_units": len(y)})
            for estimand, weights in (("ATT", design.e_star), ("ATC", 1 - design.e_star)):
                value, se = _hajek_difference(y, pair.treated, weights)
                rows.append({**base, "estimand": estimand, "value": value, "se": se, "n_units": len(y)})
            for low, high in spec["trimming"]["sensitivity_bounds"]:
                keep = (design.e_star >= low) & (design.e_star <= high)
                trimmed = neyman_difference(y[keep & pair.treated], y[keep & ~pair.treated])
                rows.append(
                    {
                        **base,
                        "estimand": f"ATE_trimmed[{low:.2f},{high:.2f}]",
                        "value": trimmed.estimate,
                        "se": trimmed.se,
                        "n_units": int(keep.sum()),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Construction checks across the Monte Carlo draws
# --------------------------------------------------------------------------------------------------


def oracle_ipw(y: np.ndarray, treated: np.ndarray, e_star: np.ndarray, ci_level: float) -> dict:
    """Hajek IPW with the TRUE design propensity: a validity check on the construction, not an estimator
    an analyst could use. Influence-function standard error; Kish effective sample size per arm."""
    weights = np.where(treated, 1 / e_star, 1 / (1 - e_star))
    estimate, se = _hajek_difference(y, treated, weights)
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    w1, w0 = weights[treated], weights[~treated]
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "ess_treated": float(w1.sum() ** 2 / (w1**2).sum()),
        "ess_control": float(w0.sum() ** 2 / (w0**2).sum()),
    }


def _logit_recovery(terms: pd.DataFrame, treated: np.ndarray) -> dict:
    fit = sm.Logit(treated.astype(float), sm.add_constant(terms.to_numpy())).fit(disp=0)
    return dict(zip(["intercept", *terms.columns], fit.params.tolist(), strict=True))


@dataclass
class ConstructionStudy:
    guard: pd.DataFrame
    benchmarks: pd.DataFrame
    draws: pd.DataFrame
    estimates: pd.DataFrame
    balance: pd.DataFrame
    pairs: dict[str, Pair] = field(repr=False)


def run_construction_study(
    processed: pd.DataFrame, spec: dict | None = None, n_draws: int | None = None
) -> ConstructionStudy:
    """Build every scenario, check the construction over all draws, and compute naive and oracle estimates.

    Scenario C retains exactly the units of Scenario A (shared uniforms, same k) and differs only in
    what the analyst may adjust for, so the naive and oracle estimates are computed once for A and B.
    """
    spec = spec or load_benchmark_config(1)
    base_seed = spec["randomness"]["base_seed"]
    n_draws = n_draws or spec["randomness"]["n_draws"]
    ci_level = spec["estimation"]["ci_level"]
    outcomes = [spec["outcomes"]["primary"], *spec["outcomes"]["secondary"]]
    pairs = {c["name"]: build_pair(processed, c, spec) for c in spec["comparisons"]}

    guard_rows = [row for pair in pairs.values() for row in overlap_guard(pair, spec)]
    benchmarks = pd.concat(
        [experimental_benchmarks(pair, spec) for pair in pairs.values()], ignore_index=True
    )

    draw_rows, estimate_rows, balance_rows = [], [], []
    for pair in pairs.values():
        terms = selection_terms(pair.frame)
        y_all = pair.frame[outcomes].to_numpy(dtype=float)
        for scenario in ("A", "B"):
            design = pair.designs[scenario]
            for draw in range(n_draws):
                keep = pair.retained(scenario, draw, base_seed)
                t = pair.treated[keep]
                x_kept = pair.covariates[keep]
                preservation = [
                    standardized_mean_difference(x_kept[c], pair.covariates[c]) for c in COVARIATE_COLUMNS
                ]
                imbalance = {
                    c: standardized_mean_difference(x_kept.loc[t, c], x_kept.loc[~t, c])
                    for c in COVARIATE_COLUMNS
                }
                recovered = _logit_recovery(terms[keep], t)
                draw_rows.append(
                    {
                        "comparison": pair.name,
                        "scenario": scenario,
                        "draw": draw,
                        "n_retained": int(keep.sum()),
                        "retention": float(keep.mean()),
                        "treated_share": float(t.mean()),
                        "max_abs_smd_vs_full_pair": float(np.max(np.abs(preservation))),
                        "max_abs_smd_treated_vs_control": float(max(abs(v) for v in imbalance.values())),
                        **{f"recovered_{k}": v for k, v in recovered.items()},
                    }
                )
                balance_rows += [
                    {"comparison": pair.name, "scenario": scenario, "draw": draw, "covariate": c, "smd": v}
                    for c, v in imbalance.items()
                ]
                y = y_all[keep]
                for j, outcome in enumerate(outcomes):
                    naive = neyman_difference(y[t, j], y[~t, j], ci_level)
                    oracle = oracle_ipw(y[:, j], t, design.e_star[keep], ci_level)
                    common = {"comparison": pair.name, "scenario": scenario, "draw": draw, "outcome": outcome}
                    estimate_rows.append(
                        {
                            **common,
                            "estimator": "naive_difference",
                            "estimand": "unadjusted association",
                            "benchmark": "ATE",
                            "estimate": naive.estimate,
                            "se": naive.se,
                            "ci_low": naive.ci_low,
                            "ci_high": naive.ci_high,
                        }
                    )
                    estimate_rows.append(
                        {**common, "estimator": "oracle_ipw", "estimand": "ATE", "benchmark": "ATE", **oracle}
                    )

    return ConstructionStudy(
        pd.DataFrame(guard_rows),
        benchmarks,
        pd.DataFrame(draw_rows),
        pd.DataFrame(estimate_rows),
        pd.DataFrame(balance_rows),
        pairs,
    )


def monte_carlo_summary(estimates: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    """Bias, absolute bias, RMSE, coverage, variance and failure rate against estimand-matched benchmarks.

    Coverage is measured against the full-pair (finite-population) benchmark. Each retained sample is a
    subset of the same trial, so estimates vary less around that benchmark than their nominal standard
    errors imply; `se_to_sd_ratio` reports this directly, and the oracle row is the calibration reference.
    """
    targets = benchmarks.rename(columns={"estimand": "benchmark", "value": "benchmark_value"})[
        ["comparison", "scenario", "outcome", "benchmark", "benchmark_value"]
    ]
    merged = estimates.merge(targets, on=["comparison", "scenario", "outcome", "benchmark"], how="left")
    merged["error"] = merged["estimate"] - merged["benchmark_value"]
    merged["covered"] = (merged["ci_low"] <= merged["benchmark_value"]) & (
        merged["benchmark_value"] <= merged["ci_high"]
    )
    merged["failed"] = ~np.isfinite(merged["estimate"]) | ~np.isfinite(merged["se"])
    keys = ["comparison", "scenario", "outcome", "estimator", "estimand", "benchmark"]
    rows = []
    for key, group in merged.groupby(keys, sort=False):
        ok = group[~group["failed"]]
        sd = ok["estimate"].std(ddof=1)
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "n_draws": len(group),
                "benchmark_value": group["benchmark_value"].iloc[0],
                "mean_estimate": ok["estimate"].mean(),
                "bias": ok["error"].mean(),
                "bias_mc_se": sd / np.sqrt(len(ok)),
                "absolute_bias": ok["error"].abs().mean(),
                "rmse": float(np.sqrt((ok["error"] ** 2).mean())),
                "empirical_sd": sd,
                "estimator_variance": sd**2,
                "mean_se": ok["se"].mean(),
                "se_to_sd_ratio": ok["se"].mean() / sd,
                "coverage": ok["covered"].mean(),
                "failure_rate": group["failed"].mean(),
            }
        )
        if "ess_treated" in group:
            row["mean_ess_treated"] = ok["ess_treated"].mean()
            row["mean_ess_control"] = ok["ess_control"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def naive_bias_decomposition(pair: Pair, spec: dict) -> pd.DataFrame:
    """Exact finite-population decomposition of what the naive comparison targets in a retained sample.

    Retained treated units are weighted by e*(x) and retained controls by 1 - e*(x), so the naive
    difference targets E_e*[Y | T=1] - E_(1-e*)[Y | T=0] over the full randomized pair. Relative to the
    ATE this equals a baseline (selection) bias, the difference in untreated outcomes between the kinds
    of customers who end up treated and untreated, plus an effect-heterogeneity term, ATT - ATE.
    """
    outcomes = [spec["outcomes"]["primary"], *spec["outcomes"]["secondary"]]
    t, c = pair.treated, ~pair.treated
    rows = []
    for name, design in pair.designs.items():
        e = design.e_star
        for outcome in outcomes:
            y = pair.frame[outcome].to_numpy(dtype=float)
            ate = y[t].mean() - y[c].mean()
            att, _ = _hajek_difference(y, t, e)
            naive_target = np.average(y[t], weights=e[t]) - np.average(y[c], weights=1 - e[c])
            baseline = np.average(y[c], weights=e[c]) - np.average(y[c], weights=1 - e[c])
            rows.append(
                {
                    "comparison": pair.name,
                    "scenario": name,
                    "outcome": outcome,
                    "ate": ate,
                    "att": att,
                    "naive_target": naive_target,
                    "bias_vs_ate": naive_target - ate,
                    "baseline_bias": baseline,
                    "heterogeneity_term": att - ate,
                }
            )
    return pd.DataFrame(rows)
