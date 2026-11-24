"""Validation extension v1 (configs/extension/v1.toml): the Part II checks on a second randomized dataset."""

from __future__ import annotations

import json
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from econml.dr import LinearDRLearner
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from causeway.balance import standardized_mean_difference
from causeway.config import CONFIG_DIR, DataPaths
from causeway.data import sha256_file
from causeway.dml import doubleml_irm, hand_dml
from causeway.estimators import (
    ArmwiseRegression,
    cross_fit_folds,
    cross_fitted_scores,
    linear_propensity_model,
    outcome_regression,
)
from causeway.experiment import lin_adjusted, neyman_difference
from causeway.propensity import fit_logistic_propensity
from causeway.weighting import Estimate, ate_weights, att_weights, hajek_difference, kish_ess

PROTOCOL = CONFIG_DIR / "extension" / "v1.toml"
GATE = CONFIG_DIR / "extension" / "criteo_gate.toml"
POOL_FILE = "criteo_pool.parquet"
NU2_WARNING = "The estimated nu2"
ESTIMANDS = {"naive_difference": "unadjusted association"}


def load_protocol(path: Path = PROTOCOL) -> dict:
    return tomllib.loads(path.read_text())


def load_gate(path: Path = GATE) -> dict:
    return tomllib.loads(path.read_text())


def pool_index(n_rows: int, size: int, seed: int) -> np.ndarray:
    """The same uniform subsample the gate used."""
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=size, replace=False))


def build_pool(paths: DataPaths | None = None) -> pd.DataFrame:
    """Verify the raw file, draw the pool and save it with its outcomes (exposure dropped)."""
    paths = paths or DataPaths()
    gate, protocol = load_gate(), load_protocol()
    raw = paths.raw / "criteo" / gate["source"]["file"]
    if not raw.exists():
        raise FileNotFoundError(
            f"{raw} not found; download it as described in configs/extension/criteo_gate.toml"
        )
    digest = sha256_file(raw)
    if digest != gate["source"]["sha256"]:
        raise ValueError(f"{raw.name} has SHA-256 {digest}, expected {gate['source']['sha256']}")
    schema = gate["schema"]
    columns = [*schema["features"], schema["treatment"], *schema["outcomes"]]
    frame = pd.read_csv(raw, usecols=columns)
    if len(frame) != schema["rows"]:
        raise ValueError(f"expected {schema['rows']:,} rows, found {len(frame):,}")
    index = pool_index(len(frame), protocol["data"]["pool_rows"], protocol["data"]["pool_seed"])
    pool = frame.iloc[index][columns].reset_index(drop=True)
    pool[schema["treatment"]] = pool[schema["treatment"]].astype(bool)
    paths.processed.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(paths.processed / POOL_FILE, index=False)
    manifest = {
        "raw_sha256": digest,
        "raw_rows": len(frame),
        "pool_rows": len(pool),
        "pool_seed": protocol["data"]["pool_seed"],
        "pool_sha256": sha256_file(paths.processed / POOL_FILE),
        "treatment_share": float(pool[schema["treatment"]].mean()),
    }
    (paths.processed / "criteo_pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return pool


def load_pool(paths: DataPaths | None = None) -> pd.DataFrame:
    paths = paths or DataPaths()
    return pd.read_parquet(paths.processed / POOL_FILE)


# Selection


def _z(column: pd.Series) -> pd.Series:
    return (column - column.mean()) / column.std(ddof=0)


def selection_index(frame: pd.DataFrame, name: str) -> np.ndarray:
    """The predeclared indices of configs/extension/v1.toml, scaled to unit SD over the pool."""
    if name == "linear":
        index = (
            0.75 * _z(frame["f0"]) - 0.75 * _z(frame["f6"]) + 0.5 * _z(frame["f9"]) + 0.5 * _z(frame["f2"])
        )
    elif name == "nonlinear":
        index = (
            1.0 * (frame["f0"] >= frame["f0"].median()).astype(float)
            - 0.5 * _z(frame["f6"]) ** 2
            + 0.6 * _z(frame["f9"]) * (frame["f2"] > frame["f2"].median()).astype(float)
            + 0.4 * _z(frame["f3"]).abs()
        )
    else:
        raise ValueError(f"unknown selection index {name!r}")
    index = index.to_numpy(dtype=float)
    return index / index.std(ddof=0)


def solve_intercept(index: np.ndarray, k: float, target: float, tolerance: float = 1e-12) -> float:
    """Intercept making the mean of expit(alpha + k * index) equal the target, by bisection."""
    low, high = -50.0, 50.0
    while high - low > tolerance:
        middle = (low + high) / 2
        if expit(middle + k * index).mean() < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


@dataclass
class Design:
    scenario: str
    e_star: np.ndarray
    keep_probability: np.ndarray  # acceptance probability of each pool unit
    hidden: list[str]


def make_designs(pool: pd.DataFrame, protocol: dict) -> dict[str, Design]:
    t = pool["treatment"].to_numpy(dtype=bool)
    share = t.mean()
    designs = {}
    for name, scenario in protocol["scenarios"].items():
        index = selection_index(pool, scenario["index"])
        alpha = solve_intercept(index, scenario["k"], share)
        e_star = expit(alpha + scenario["k"] * index)
        ratio = np.where(t, e_star / share, (1 - e_star) / (1 - share))
        designs[name] = Design(name, e_star, ratio / ratio.max(), list(scenario["hidden"]))
    return designs


def draw_seeds(base_seed: int, draw: int) -> tuple[np.random.Generator, int]:
    """Selection generator (shared by every scenario) and the estimator seed for one draw."""
    selection = np.random.default_rng(np.random.SeedSequence(entropy=base_seed, spawn_key=(draw, 0)))
    estimator = int(np.random.SeedSequence(entropy=base_seed, spawn_key=(draw, 1)).generate_state(1)[0])
    return selection, estimator


# References


def references(pool: pd.DataFrame, designs: dict[str, Design], protocol: dict) -> pd.DataFrame:
    """Every reference, computed on the full pool, with its standard error."""
    t = pool["treatment"].to_numpy(dtype=bool)
    x = pool[protocol["analyst"]["covariates"]]
    low, high = protocol["references"]["ATE_trimmed"]["bounds"]
    outcomes = [protocol["outcomes"]["primary"], *protocol["outcomes"]["secondary"]]
    rows = []
    for outcome in outcomes:
        y = pool[outcome].to_numpy(dtype=float)
        ate = neyman_difference(y[t], y[~t])
        adjusted = lin_adjusted(y, t, x)
        for name, design in designs.items():
            inside = (design.e_star >= low) & (design.e_star <= high)
            att = hajek_difference(y, t, design.e_star)
            trimmed = neyman_difference(y[inside & t], y[inside & ~t])
            for estimand, value in (
                ("ATE", ate),
                ("ATE_adjusted", adjusted),
                ("ATT", att),
                ("ATE_trimmed", trimmed),
            ):
                rows.append(
                    {
                        "comparison": "criteo",
                        "scenario": name,
                        "outcome": outcome,
                        "estimand": estimand,
                        "value": value.estimate,
                        "se": value.se,
                        "ci_low": value.ci_low,
                        "ci_high": value.ci_high,
                        "share_of_pool": float(inside.mean()) if estimand == "ATE_trimmed" else 1.0,
                    }
                )
    return pd.DataFrame(rows)


# Estimators on one constructed sample


def _counting_nu2(function, *args, **kwargs):
    """Call a DoubleML fit, count nonpositive-nu2 fallbacks, and re-emit every warning unchanged."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args, **kwargs)
    for w in caught:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return result, sum(str(w.message).startswith(NU2_WARNING) for w in caught)


def _max_abs_smd(x: pd.DataFrame, t: np.ndarray, weights: np.ndarray | None = None) -> float:
    out = 0.0
    for column in x.columns:
        values = x[column].to_numpy(dtype=float)
        wt = None if weights is None else weights[t]
        wc = None if weights is None else weights[~t]
        out = max(out, abs(standardized_mean_difference(values[t], values[~t], wt, wc)))
    return out


def analyse_sample(
    x_full: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    e_star: np.ndarray,
    hidden: list[str],
    outcomes: list[str],
    protocol: dict,
    seed: int,
    cross_check: bool,
) -> tuple[list[dict], dict]:
    """Every estimator of the protocol on one retained sample."""
    ci = protocol["estimation"]["ci_level"]
    low, high = protocol["references"]["ATE_trimmed"]["bounds"]
    x = x_full.drop(columns=hidden)
    rows = []

    def add(estimator: str, benchmark: str, estimates: list):
        for outcome, estimate in zip(outcomes, estimates, strict=True):
            rows.append(
                {
                    "estimator": estimator,
                    "estimand": ESTIMANDS.get(estimator, benchmark),
                    "benchmark": benchmark,
                    "outcome": outcome,
                    "estimate": estimate.estimate,
                    "se": estimate.se,
                    "ci_low": estimate.ci_low,
                    "ci_high": estimate.ci_high,
                }
            )

    columns = [y[:, j] for j in range(y.shape[1])]
    add("naive_difference", "ATE", [neyman_difference(c[t], c[~t], ci) for c in columns])
    add("oracle_ipw", "ATE", [hajek_difference(c, t, ate_weights(e_star, t), ci) for c in columns])
    add("oracle_ipw_att", "ATT", [hajek_difference(c, t, att_weights(e_star, t), ci) for c in columns])
    inside_true = (e_star >= low) & (e_star <= high)
    add(
        "oracle_trimmed",
        "ATE_trimmed",
        [
            hajek_difference(c[inside_true], t[inside_true], ate_weights(e_star, t)[inside_true], ci)
            for c in columns
        ],
    )

    scores = fit_logistic_propensity(x, t).scores
    w_ate, w_att = ate_weights(scores, t), att_weights(scores, t)
    add("ipw_ate", "ATE", [hajek_difference(c, t, w_ate, ci) for c in columns])
    add("ipw_att", "ATT", [hajek_difference(c, t, w_att, ci) for c in columns])
    add("outcome_regression", "ATE", [outcome_regression(c, t, x, ci) for c in columns])

    folds = cross_fit_folds(len(t), protocol["estimation"]["cross_fit_folds"], seed)
    fit = cross_fitted_scores(y, t, x, folds, linear_propensity_model, LinearRegression)
    add("aipw", "ATE", fit.estimates(ci))
    inside_estimated = (fit.propensity >= low) & (fit.propensity <= high)
    add("aipw_trimmed", "ATE_trimmed", fit.estimates(ci, inside_estimated))

    # Each DoubleML fit fails on its own: DoubleML refuses, for example, when a regressor's predictions
    # for a binary outcome come out exactly 0 or 1. A failed fit is a failed row, not a failed sample.
    fallbacks, dml_errors, dml, propensity = 0, [], {"ATE": [], "ATTE": []}, None
    for c in columns:
        for score, fitted in dml.items():
            try:
                (estimate, predictions), count = _counting_nu2(
                    doubleml_irm, c, t, x, folds, seed, ci, score=score
                )
            except ValueError as error:
                estimate, count, predictions = Estimate(np.nan, np.nan, np.nan, np.nan), 0, None
                dml_errors.append(f"{score}: {str(error)[:120]}")
            fitted.append(estimate)
            fallbacks += count
            if propensity is None and predictions is not None:
                propensity = predictions["ml_m"]
    add("dml_irm", "ATE", dml["ATE"])
    add("dml_irm_att", "ATT", dml["ATTE"])

    if cross_check:
        econml = []
        for c in columns:
            learner = LinearDRLearner(
                model_propensity=linear_propensity_model(),
                model_regression=ArmwiseRegression(),
                cv=folds,
                min_propensity=1e-12,
                random_state=seed,
            )
            learner.fit(c, t.astype(int), X=None, W=x.to_numpy())
            lo, hi = learner.ate_interval(alpha=1 - ci)
            econml.append(Estimate(float(learner.ate()), np.nan, float(lo), float(hi)))
        add("aipw_econml", "ATE", econml)
        add("dml_hand", "ATE", hand_dml(y, t, x, folds, seed).estimates(ci))

    weights_t, weights_c = w_ate[t], w_ate[~t]
    diagnostics = {
        "n": len(t),
        "n_treated": int(t.sum()),
        "max_abs_smd_unweighted": _max_abs_smd(x, t),
        "max_abs_smd_ipw": _max_abs_smd(x, t, w_ate),
        "max_abs_smd_hidden_unweighted": _max_abs_smd(x_full[hidden], t) if hidden else np.nan,
        "max_abs_smd_hidden_ipw": _max_abs_smd(x_full[hidden], t, w_ate) if hidden else np.nan,
        "ess_treated": kish_ess(weights_t) / len(weights_t),
        "ess_control": kish_ess(weights_c) / len(weights_c),
        "max_weight_ratio_control": float(weights_c.max() / weights_c.mean()),
        "share_propensity_outside_01_99": float(((scores < 0.01) | (scores > 0.99)).mean()),
        "share_true_propensity_outside_01_99": float(((e_star < 0.01) | (e_star > 0.99)).mean()),
        "dml_propensity_min": float(propensity.min()) if propensity is not None else np.nan,
        "dml_propensity_max": float(propensity.max()) if propensity is not None else np.nan,
        "dml_fits": 2 * len(columns),
        "dml_failed_fits": len(dml_errors),
        "dml_error": " | ".join(dml_errors),
        "nu2_fallbacks": fallbacks,
        "trimmed_share_estimated": float(inside_estimated.mean()),
        "trimmed_share_true": float(inside_true.mean()),
        "trimmed_set_agreement": float((inside_estimated == inside_true).mean()),
    }
    return rows, diagnostics


def _run_draw(draw, pool_x, t_pool, y_pool, designs, outcomes, protocol, cross_check_draws):
    selection, seed = draw_seeds(protocol["randomness"]["base_seed"], draw)
    uniforms = selection.random(len(t_pool))
    estimates, diagnostics = [], []
    with threadpool_limits(limits=1):
        for name, design in designs.items():
            keep = uniforms <= design.keep_probability
            base = {"comparison": "criteo", "scenario": name, "draw": draw}
            try:
                rows, extra = analyse_sample(
                    pool_x[keep].reset_index(drop=True),
                    t_pool[keep],
                    y_pool[keep],
                    design.e_star[keep],
                    design.hidden,
                    outcomes,
                    protocol,
                    seed,
                    draw < cross_check_draws,
                )
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as error:
                # A failed sample still counts towards every estimator's failure rate.
                rows = [
                    {
                        "estimator": e["name"],
                        "estimand": e["estimand"],
                        "benchmark": e["reference"],
                        "outcome": outcome,
                        "estimate": np.nan,
                        "se": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                    }
                    for e in protocol["estimators"]
                    for outcome in outcomes
                ]
                extra = {"error": f"{type(error).__name__}: {error}"}
            estimates += [{**base, **row} for row in rows]
            diagnostics.append({**base, **extra})
    return estimates, diagnostics


def run_study(
    pool: pd.DataFrame, protocol: dict | None = None, n_draws: int | None = None, n_jobs: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """References, estimates and per-sample diagnostics for every scenario and draw."""
    protocol = protocol or load_protocol()
    n_draws = n_draws or protocol["randomness"]["n_draws"]
    outcomes = [protocol["outcomes"]["primary"], *protocol["outcomes"]["secondary"]]
    designs = make_designs(pool, protocol)
    pool_x = pool[protocol["analyst"]["covariates"]]
    t = pool["treatment"].to_numpy(dtype=bool)
    y = pool[outcomes].to_numpy(dtype=float)
    cross_check_draws = protocol["cross_checks"]["draws"]
    args = (pool_x, t, y, designs, outcomes, protocol, cross_check_draws)
    if n_jobs == 1:
        results = [_run_draw(draw, *args) for draw in range(n_draws)]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_run_draw)(draw, *args) for draw in range(n_draws))
    estimates = pd.DataFrame([row for rows, _ in results for row in rows])
    diagnostics = pd.DataFrame([row for _, rows in results for row in rows])
    keys = ["scenario", "draw"]
    estimates = estimates.sort_values([*keys, "estimator", "outcome"], kind="stable").reset_index(drop=True)
    diagnostics = diagnostics.sort_values(keys, kind="stable").reset_index(drop=True)
    return references(pool, designs, protocol), estimates, diagnostics


def design_summary(pool: pd.DataFrame, protocol: dict | None = None) -> pd.DataFrame:
    """Overlap and selection strength of each scenario, from the covariates and e* alone."""
    protocol = protocol or load_protocol()
    x = pool[protocol["analyst"]["covariates"]]
    rows = []
    for name, design in make_designs(pool, protocol).items():
        e = design.e_star
        rows.append(
            {
                "scenario": name,
                "label": protocol["scenarios"][name]["label"],
                "e_star_p01": float(np.quantile(e, 0.01)),
                "e_star_p50": float(np.median(e)),
                "e_star_p99": float(np.quantile(e, 0.99)),
                "share_e_star_above_0_99": float((e > 0.99).mean()),
                "share_e_star_below_0_01": float((e < 0.01).mean()),
                "expected_retained": float(design.keep_probability.sum()),
                "max_abs_smd_expected": max(
                    abs(np.average(x[c], weights=e) - np.average(x[c], weights=1 - e)) / x[c].std(ddof=0)
                    for c in x.columns
                ),
                "hidden": ", ".join(design.hidden) or "none",
            }
        )
    return pd.DataFrame(rows)


# Summaries and report

CROSS_CHECKS = {"aipw_econml": "aipw", "dml_hand": "dml_irm"}
ORACLES = {"ATE": "oracle_ipw", "ATT": "oracle_ipw_att", "ATE_trimmed": "oracle_trimmed"}


def summarize(refs: pd.DataFrame, estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monte Carlo summary against each estimator's own reference, and, for ATE estimators, against the
    covariate-adjusted reference as well."""
    from causeway.benchmark import monte_carlo_summary

    main = estimates[~estimates["estimator"].isin(CROSS_CHECKS)]
    targets = refs[["comparison", "scenario", "outcome", "estimand", "value"]]
    summary = monte_carlo_summary(main, targets)
    oracle = summary[summary["estimator"].isin(ORACLES.values())].set_index(
        ["scenario", "outcome", "benchmark"]
    )
    summary["oracle_coverage"] = [
        oracle["coverage"].get((s, o, b), np.nan)
        for s, o, b in summary[["scenario", "outcome", "benchmark"]].itertuples(index=False)
    ]
    summary["bias_pct"] = summary["bias"] / summary["benchmark_value"]
    summary["recovers"] = summary["bias"].abs() <= 2 * summary["bias_mc_se"]

    adjusted = targets[targets["estimand"] == "ATE_adjusted"].assign(estimand="ATE")
    versus_adjusted = monte_carlo_summary(main[main["benchmark"] == "ATE"], adjusted)
    versus_adjusted["bias_pct"] = versus_adjusted["bias"] / versus_adjusted["benchmark_value"]
    versus_adjusted["recovers"] = versus_adjusted["bias"].abs() <= 2 * versus_adjusted["bias_mc_se"]
    return summary, versus_adjusted


def cross_check_table(estimates: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "draw", "outcome"]
    rows = []
    for check, primary in CROSS_CHECKS.items():
        a = estimates[estimates["estimator"] == check].set_index(keys)["estimate"]
        b = estimates[estimates["estimator"] == primary].set_index(keys)["estimate"]
        joined = pd.concat([a.rename("check"), b.rename("primary")], axis=1, join="inner")
        for scenario, group in joined.groupby(level="scenario"):
            rows.append(
                {
                    "check": f"{check} vs {primary}",
                    "scenario": scenario,
                    "comparisons": len(group),
                    "max_abs_difference": float((group["check"] - group["primary"]).abs().max()),
                }
            )
    return pd.DataFrame(rows)


def extension_provenance(paths: DataPaths) -> dict:
    from causeway.pipeline import environment
    from causeway.provenance import _git_state, _sha256_lf

    manifest = json.loads((paths.processed / "criteo_pool_manifest.json").read_text())
    return {
        "data": manifest,
        "protocol": {"path": "configs/extension/v1.toml", "sha256": _sha256_lf(PROTOCOL)},
        "gate": {"path": "configs/extension/criteo_gate.toml", "sha256": _sha256_lf(GATE)},
        "code": _git_state(),
        "environment": environment(),
    }


EXTENSION_TABLES = [
    "references",
    "design",
    "estimates",
    "diagnostics",
    "summary",
    "summary_vs_adjusted",
    "cross_checks",
]


def compute_extension(paths: DataPaths | None = None, n_jobs: int = 8) -> dict[str, pd.DataFrame]:
    paths = paths or DataPaths()
    protocol = load_protocol()
    pool = load_pool(paths)
    refs, estimates, diagnostics = run_study(pool, protocol, n_jobs=n_jobs)
    summary, versus_adjusted = summarize(refs, estimates)
    tables = {
        "references": refs,
        "design": design_summary(pool, protocol),
        "estimates": estimates,
        "diagnostics": diagnostics,
        "summary": summary,
        "summary_vs_adjusted": versus_adjusted,
        "cross_checks": cross_check_table(estimates),
    }
    results = paths.results("extension")
    results.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(results / f"{name}.csv", index=False)
    (results / "provenance.json").write_text(json.dumps(extension_provenance(paths), indent=2) + "\n")
    return tables


FIGURE_ESTIMATORS = [
    "naive_difference",
    "oracle_ipw",
    "oracle_ipw_att",
    "oracle_trimmed",
    "outcome_regression",
    "ipw_ate",
    "ipw_att",
    "aipw",
    "aipw_trimmed",
    "dml_irm",
    "dml_irm_att",
]


def _md(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def _pct(value: float, recovers: bool | None = None) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:+.1%}" + ("" if recovers in (None, True) else " *")


def _ci(row: pd.Series, scale: float) -> str:
    return f"[{row['ci_low'] * scale:.2f}, {row['ci_high'] * scale:.2f}]"


def extension_markdown(tables: dict[str, pd.DataFrame], protocol: dict, provenance: dict) -> str:
    refs, summary, adjusted = tables["references"], tables["summary"], tables["summary_vs_adjusted"]
    diagnostics, design, checks = tables["diagnostics"], tables["design"], tables["cross_checks"]
    outcomes = [protocol["outcomes"]["primary"], *protocol["outcomes"]["secondary"]]
    scenarios = list(protocol["scenarios"])
    feasible = [e for e in FIGURE_ESTIMATORS if not e.startswith(("oracle", "naive"))]
    data = provenance["data"]

    lines = [
        "# CauseWay validation extension v1: Criteo Uplift v2.1",
        "",
        "A separately versioned check of the Part II conclusions on a second randomized dataset, which "
        "pools several advertising tests, with linear, nonlinear, "
        "poor-overlap and hidden-confounder selection. Protocol: `configs/extension/v1.toml`, "
        "frozen before any outcome of this dataset was read; gate: `configs/extension/criteo_gate.toml`, "
        "written before its statistics were computed. Commit order records that sequence; it cannot by "
        "itself prove what was "
        "inspected. The original study and its results are unchanged.",
        "",
        "Effects are in percentage points (pp) of the visit or conversion rate.",
        "",
        "## 1. Data and references",
        "",
        f"Pool: {data['pool_rows']:,} users drawn uniformly (seed {data['pool_seed']}) from the "
        f"{data['raw_rows']:,}-row file (SHA-256 `{data['raw_sha256'][:12]}…`), treatment share "
        f"{data['treatment_share']:.3f}. At the gate a classifier predicted treatment from the features "
        f"with out-of-fold AUC {protocol['data']['gate_out_of_fold_auc']:.3f}, and the largest "
        f"arm difference was |SMD| {protocol['data']['gate_max_abs_smd']:.3f}: small but real imbalance.",
        "",
        "Every reference is computed on the full pool. Conditional on the pool it is the exact target of the "
        "construction; as an estimate of a population effect it has the standard error shown.",
        "",
    ]
    rows = []
    for outcome in outcomes:
        r = refs[(refs["outcome"] == outcome) & (refs["scenario"] == scenarios[0])].set_index("estimand")
        for estimand in ("ATE", "ATE_adjusted"):
            row = r.loc[estimand]
            rows.append(
                {
                    "Outcome": outcome,
                    "Reference": "difference in means" if estimand == "ATE" else "regression-adjusted",
                    "Value (pp)": f"{row['value'] * 100:.3f}",
                    "SE (pp)": f"{row['se'] * 100:.3f}",
                    "95% CI (pp)": _ci(row, 100),
                }
            )
        gap = r.loc["ATE_adjusted", "value"] - r.loc["ATE", "value"]
        rows.append(
            {
                "Outcome": outcome,
                "Reference": "gap (adjusted − difference)",
                "Value (pp)": f"{gap * 100:+.3f}",
                "SE (pp)": "",
                "95% CI (pp)": f"{gap / r.loc['ATE', 'se']:+.2f} SE of the difference in means",
            }
        )
    lines += [_md(pd.DataFrame(rows)), ""]
    rows = []
    for scenario in scenarios:
        for outcome in outcomes:
            r = refs[(refs["outcome"] == outcome) & (refs["scenario"] == scenario)].set_index("estimand")
            rows.append(
                {
                    "Scenario": scenario,
                    "Outcome": outcome,
                    "ATT (pp)": f"{r.loc['ATT', 'value'] * 100:.3f} ± {1.96 * r.loc['ATT', 'se'] * 100:.3f}",
                    "Trimmed ATE (pp)": f"{r.loc['ATE_trimmed', 'value'] * 100:.3f} ± "
                    f"{1.96 * r.loc['ATE_trimmed', 'se'] * 100:.3f}",
                    "Trimmed share of pool": f"{r.loc['ATE_trimmed', 'share_of_pool']:.0%}",
                }
            )
    lines += ["ATT and trimmed references depend on the scenario's e*(x):", "", _md(pd.DataFrame(rows)), ""]

    lines += ["## 2. Scenarios", ""]
    lines += [
        _md(
            design.assign(
                **{
                    "e* 1st–99th pct": [
                        f"{a:.2f}–{b:.2f}"
                        for a, b in zip(design["e_star_p01"], design["e_star_p99"], strict=True)
                    ],
                    "e* > 0.99": design["share_e_star_above_0_99"].map("{:.1%}".format),
                    "expected retained": design["expected_retained"].map("{:,.0f}".format),
                    "largest arm SMD": design["max_abs_smd_expected"].map("{:.2f}".format),
                }
            )[
                [
                    "scenario",
                    "label",
                    "e* 1st–99th pct",
                    "e* > 0.99",
                    "expected retained",
                    "largest arm SMD",
                    "hidden",
                ]
            ]
        ),
        "",
    ]

    lines += [
        "## 3. Estimator recovery",
        "",
        "![Recovery heatmap](../../figures/extension/extension_recovery.png)",
        "",
        "Mean bias over 200 draws as a share of each estimator's own reference; `*` marks a bias larger than "
        "twice its Monte Carlo SE. Each cell is a separate test at about the 5% level, so isolated flags are "
        "weak "
        "evidence. Coverage is read against the oracle's, since every draw comes from the same pool.",
        "",
    ]
    for outcome in outcomes:
        cells = summary[summary["outcome"] == outcome].set_index(["estimator", "scenario"])
        rows = []
        for estimator in FIGURE_ESTIMATORS:
            row = {"Estimator": estimator, "Reference": cells.loc[(estimator, scenarios[0]), "benchmark"]}
            for scenario in scenarios:
                c = cells.loc[(estimator, scenario)]
                row[f"{scenario} bias"] = _pct(c["bias_pct"], c["recovers"])
                row[f"{scenario} cover"] = "—" if not np.isfinite(c["coverage"]) else f"{c['coverage']:.0%}"
            row["max failure rate"] = f"{cells.loc[estimator]['failure_rate'].max():.1%}"
            rows.append(row)
        lines += [f"**{outcome.capitalize()}**", "", _md(pd.DataFrame(rows)), ""]

    lines += ["Feasible estimators whose bias is within Monte Carlo error, by scenario:", ""]
    for scenario in scenarios:
        parts = []
        for outcome in outcomes:
            cells = summary[(summary["outcome"] == outcome) & (summary["scenario"] == scenario)].set_index(
                "estimator"
            )
            ok = [e for e in feasible if bool(cells.loc[e, "recovers"]) and cells.loc[e, "failure_rate"] < 1]
            parts.append(f"{outcome}: {', '.join(ok) or 'none'}")
        lines.append(
            f"- **{scenario}** ({protocol['scenarios'][scenario]['label'].lower()}): "
            + "; ".join(parts)
            + "."
        )
    lines.append("")

    lines += [
        "## 4. Against the regression-adjusted reference",
        "",
        "ATE estimators adjust for the features, so if the pool's slight imbalance mattered they would track "
        "the "
        "adjusted reference rather than the difference in means. Bias against the adjusted reference:",
        "",
    ]
    rows = []
    for outcome in outcomes:
        for estimator in ["oracle_ipw", "outcome_regression", "ipw_ate", "aipw", "dml_irm"]:
            row = {"Outcome": outcome, "Estimator": estimator}
            for scenario in scenarios:
                c = adjusted[
                    (adjusted["outcome"] == outcome)
                    & (adjusted["estimator"] == estimator)
                    & (adjusted["scenario"] == scenario)
                ].iloc[0]
                row[scenario] = _pct(c["bias_pct"], c["recovers"])
            rows.append(row)
    lines += [_md(pd.DataFrame(rows)), ""]

    lines += ["## 5. Balance, overlap and fit diagnostics", "", "Means over draws."]
    d = (
        diagnostics.groupby("scenario")
        .agg(
            n=("n", "mean"),
            smd_before=("max_abs_smd_unweighted", "mean"),
            smd_ipw=("max_abs_smd_ipw", "mean"),
            smd_hidden_before=("max_abs_smd_hidden_unweighted", "mean"),
            smd_hidden_ipw=("max_abs_smd_hidden_ipw", "mean"),
            ess_control=("ess_control", "mean"),
            max_weight=("max_weight_ratio_control", "mean"),
            outside=("share_propensity_outside_01_99", "mean"),
            trimmed_agreement=("trimmed_set_agreement", "mean"),
            dml_fits=("dml_fits", "sum"),
            dml_failed=("dml_failed_fits", "sum"),
            nu2=("nu2_fallbacks", "sum"),
        )
        .reset_index()
    )
    d = d.set_index("scenario").loc[scenarios].reset_index()
    lines += [
        "",
        _md(
            pd.DataFrame(
                {
                    "Scenario": d["scenario"],
                    "Retained users": d["n"].map("{:,.0f}".format),
                    "Largest |SMD| before / after IPW": [
                        f"{a:.2f} / {b:.2f}" for a, b in zip(d["smd_before"], d["smd_ipw"], strict=True)
                    ],
                    "Hidden f6 |SMD| before / after": [
                        "—" if not np.isfinite(a) else f"{a:.2f} / {b:.2f}"
                        for a, b in zip(d["smd_hidden_before"], d["smd_hidden_ipw"], strict=True)
                    ],
                    "Control ESS share": d["ess_control"].map("{:.0%}".format),
                    "Largest control weight / mean": d["max_weight"].map("{:.0f}".format),
                    "ê outside [0.01, 0.99]": d["outside"].map("{:.1%}".format),
                    "Trimmed-set agreement (ê vs e*)": d["trimmed_agreement"].map("{:.1%}".format),
                    "DoubleML fits / failed / nu2 fallbacks": [
                        f"{int(a):,} / {int(b):,} / {int(c):,}"
                        for a, b, c in zip(d["dml_fits"], d["dml_failed"], d["nu2"], strict=True)
                    ],
                }
            )
        ),
        "",
        "DoubleML's nonpositive-nu2 fallbacks are recorded, not suppressed. They concern only DoubleML's own "
        "sensitivity analysis, which is not used; the estimates and SEs come from the score.",
        "",
    ]
    errors = diagnostics.get("dml_error", pd.Series(dtype=str)).fillna("")
    reasons = sorted({e.split(": ", 1)[1][:90] for msg in errors for e in msg.split(" | ") if e})
    if reasons:
        lines += ["Reasons DoubleML refused a fit: " + "; ".join(reasons) + ".", ""]

    lines += [
        "## 6. Implementation cross-checks",
        "",
        "First 10 draws of each scenario, same models and folds.",
        "",
    ]
    lines += [_md(checks.assign(max_abs_difference=checks["max_abs_difference"].map("{:.1e}".format))), ""]

    lines += _post_hoc_sections(tables)
    lines += ["## 9. Expectations written before outcomes were read", ""]
    for key, text in protocol["expectations"].items():
        lines.append(f"- **{key}**: {text}")
    lines += [
        "",
        "## 10. What this does and does not show",
        "",
        "- It tests estimator behaviour under selection rules that are known by construction. Success here "
        "does "
        "not establish identification in other observational data, where the selection rule is unknown.",
        "- Scenario H can show bias only to the extent that the hidden feature predicts outcomes given the "
        "others; that relation was unknown when the protocol was frozen.",
        "- The original study's hidden-confounder failures and negative targeting result stand as reported; "
        "this "
        "extension neither replaces nor reruns them.",
        "",
        "## 11. Provenance and commands",
        "",
        f"- Code: `{(provenance['code']['commit'] or 'unknown')[:7]}`"
        + (
            " with uncommitted changes to `src/` or `configs/`."
            if provenance["code"]["uncommitted_changes"]
            else "."
        ),
        f"- Protocol SHA-256 `{provenance['protocol']['sha256'][:12]}…`; "
        f"gate SHA-256 `{provenance['gate']['sha256'][:12]}…`.",
        f"- Pool parquet SHA-256 `{provenance['data']['pool_sha256'][:12]}…`.",
        *(
            [
                "- Post-hoc analyses (Sections 7 and 8): code "
                f"`{(provenance['post_hoc']['code']['commit'] or 'unknown')[:7]}`"
                + (
                    ", with uncommitted changes."
                    if provenance["post_hoc"]["code"]["uncommitted_changes"]
                    else "."
                )
            ]
            if "post_hoc" in provenance
            else []
        ),
        f"- Python {provenance['environment']['python']}; "
        + ", ".join(f"{k} {v}" for k, v in provenance["environment"]["packages"].items())
        + ".",
        "",
        "```bash",
        "make extension-data   # verify the downloaded file and build the pool",
        "make extension        # run the 800 constructed samples and write this report",
        "```",
        "",
    ]
    return "\n".join(lines)


# Post-hoc analyses: added after the frozen run was read, reported separately and labelled as such.

POST_HOC_SEED = 20181017
POST_HOC_TABLES = ["post_hoc_reference_checks", "post_hoc_hidden_shift"]


def reference_checks(pool: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    """The full-pool effect with and without covariate adjustment: is the difference in means a clean
    reference for estimators that adjust for the features?"""
    x = pool[protocol["analyst"]["covariates"]]
    t = pool["treatment"].to_numpy(dtype=bool)
    folds = cross_fit_folds(len(t), protocol["estimation"]["cross_fit_folds"], POST_HOC_SEED)
    rows = []
    for outcome in [protocol["outcomes"]["primary"], *protocol["outcomes"]["secondary"]]:
        y = pool[outcome].to_numpy(dtype=float)
        dim = neyman_difference(y[t], y[~t])
        aipw = cross_fitted_scores(y, t, x, folds, linear_propensity_model, LinearRegression).estimates()[0]
        dml, predictions = doubleml_irm(y, t, x, folds, POST_HOC_SEED)
        for method, value in (
            ("difference in means", dim),
            ("Lin regression adjustment", lin_adjusted(y, t, x)),
            ("cross-fitted AIPW, linear", aipw),
            ("DML, gradient boosting", dml),
        ):
            rows.append(
                {
                    "outcome": outcome,
                    "method": method,
                    "value": value.estimate,
                    "se": value.se,
                    "gap_from_difference": value.estimate - dim.estimate,
                    "gap_in_difference_se": (value.estimate - dim.estimate) / dim.se,
                    "pool_propensity_min": float(predictions["ml_m"].min()),
                    "pool_propensity_max": float(predictions["ml_m"].max()),
                }
            )
    return pd.DataFrame(rows)


def hidden_shift(estimates: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Scenario H retains exactly the users of scenario A and only hides f6, so the per-draw difference
    H - A isolates the effect of the hidden feature without relying on any reference."""
    keys = ["draw", "estimator", "outcome"]
    a = estimates[estimates["scenario"] == "A"].set_index(keys)["estimate"]
    h = estimates[estimates["scenario"] == "H"].set_index(keys)["estimate"]
    shift = (h - a).dropna().rename("shift").reset_index()
    ate = refs[(refs["scenario"] == "A") & (refs["estimand"] == "ATE")].set_index("outcome")["value"]
    rows = []
    for (estimator, outcome), group in shift.groupby(["estimator", "outcome"]):
        mean, mc_se = group["shift"].mean(), group["shift"].std(ddof=1) / np.sqrt(len(group))
        rows.append(
            {
                "estimator": estimator,
                "outcome": outcome,
                "draws": len(group),
                "mean_shift": mean,
                "mc_se": mc_se,
                "shift_pct_of_difference_in_means": mean / ate[outcome],
                "beyond_2_mc_se": bool(abs(mean) > 2 * mc_se),
            }
        )
    return pd.DataFrame(rows)


def compute_post_hoc(paths: DataPaths | None = None) -> dict[str, pd.DataFrame]:
    from causeway.pipeline import environment
    from causeway.provenance import _git_state

    paths = paths or DataPaths()
    protocol = load_protocol()
    results = paths.results("extension")
    estimates = pd.read_csv(results / "estimates.csv")
    refs = pd.read_csv(results / "references.csv")
    tables = {
        "post_hoc_reference_checks": reference_checks(load_pool(paths), protocol),
        "post_hoc_hidden_shift": hidden_shift(estimates, refs),
    }
    for name, table in tables.items():
        table.to_csv(results / f"{name}.csv", index=False)
    stamp = {"code": _git_state(), "environment": environment(), "seed": POST_HOC_SEED}
    (results / "post_hoc_provenance.json").write_text(json.dumps(stamp, indent=2) + "\n")
    return tables


def _post_hoc_sections(tables: dict[str, pd.DataFrame]) -> list[str]:
    if "post_hoc_reference_checks" not in tables:
        return []
    checks, shift = tables["post_hoc_reference_checks"], tables["post_hoc_hidden_shift"]
    summary, adjusted, refs = tables["summary"], tables["summary_vs_adjusted"], tables["references"]
    lines = [
        "## 7. Post-hoc analyses (not in the frozen protocol)",
        "",
        "Added after the frozen run had been read, because the two predeclared ATE references disagree. "
        "They use only saved outputs and the pool.",
        "",
        "**Full-pool effect with and without adjustment.** If treatment were unrelated to the features in "
        "the pool, every row would agree up to noise much smaller than the difference in means' own SE.",
        "",
        _md(
            pd.DataFrame(
                {
                    "Outcome": checks["outcome"],
                    "Method": checks["method"],
                    "Effect (pp)": (checks["value"] * 100).map("{:.3f}".format),
                    "SE (pp)": (checks["se"] * 100).map("{:.3f}".format),
                    "Gap from difference in means": [
                        "—" if m == "difference in means" else f"{g * 100:+.3f} pp ({z:+.2f} SE)"
                        for m, g, z in zip(
                            checks["method"],
                            checks["gap_from_difference"],
                            checks["gap_in_difference_se"],
                            strict=True,
                        )
                    ],
                }
            )
        ),
        "",
        f"A cross-fitted gradient-boosting propensity on the full pool ranges from "
        f"{checks['pool_propensity_min'].iloc[0]:.2f} to {checks['pool_propensity_max'].iloc[0]:.2f}.",
        "",
        "**Hiding f6, paired by draw.** Scenario H keeps exactly the users of scenario A and only hides f6, "
        "so the per-draw difference H − A measures the effect of the hidden feature without any reference.",
        "",
    ]
    shown = shift[
        shift["estimator"].isin(
            ["outcome_regression", "ipw_ate", "ipw_att", "aipw", "dml_irm", "dml_irm_att"]
        )
    ]
    lines += [
        _md(
            pd.DataFrame(
                {
                    "Outcome": shown["outcome"],
                    "Estimator": shown["estimator"],
                    "Mean shift H − A (pp)": [
                        f"{m * 100:+.3f} ± {2 * e * 100:.3f}"
                        for m, e in zip(shown["mean_shift"], shown["mc_se"], strict=True)
                    ],
                    "As % of the difference in means": shown["shift_pct_of_difference_in_means"].map(
                        "{:+.1%}".format
                    ),
                    "Beyond 2 MC SE": shown["beyond_2_mc_se"].map({True: "yes", False: "no"}),
                }
            )
        ),
        "",
    ]

    def cell(table, estimator, scenario, outcome="visit"):
        return table[
            (table["estimator"] == estimator)
            & (table["scenario"] == scenario)
            & (table["outcome"] == outcome)
        ].iloc[0]

    visit = refs[(refs["outcome"] == "visit") & (refs["scenario"] == "A")].set_index("estimand")
    gap = visit.loc["ATE_adjusted", "value"] - visit.loc["ATE", "value"]
    dml_pool = checks[(checks["outcome"] == "visit") & (checks["method"] == "DML, gradient boosting")].iloc[0]
    oracle_flags = int((~summary[summary["estimator"].str.startswith("oracle")]["recovers"]).sum())
    oracle_cells = int(summary["estimator"].str.startswith("oracle").sum())
    recover_adjusted = [
        e
        for e in ["outcome_regression", "ipw_ate", "aipw", "dml_irm"]
        if bool(cell(adjusted, e, "A")["recovers"])
    ]
    h = shift[shift["outcome"] == "visit"].set_index("estimator")
    moved = {e: h.loc[e, "mean_shift"] * 100 for e in ["ipw_ate", "aipw", "outcome_regression", "dml_irm"]}
    n_bias = {e: cell(summary, e, "N")["bias_pct"] for e in ["ipw_ate", "aipw"]}
    n_adjusted = {e: cell(adjusted, e, "N")["bias_pct"] for e in ["ipw_ate", "aipw", "dml_irm"]}
    oracle_p_coverage = cell(summary, "oracle_ipw", "P")["coverage"]
    a_bias = [cell(summary, e, "A")["bias_pct"] for e in ["outcome_regression", "ipw_ate", "aipw", "dml_irm"]]
    lines += [
        "## 8. Reading the results",
        "",
        f"- **The construction works.** {oracle_flags} of {oracle_cells} oracle cells show bias beyond twice "
        "their Monte Carlo SE.",
        f"- **The difference in means is not a clean reference for adjusted estimators on visit.** Adjusting "
        f"for the features moves the full-pool visit effect by {gap * 100:+.3f} pp "
        f"({gap / visit.loc['ATE', 'value']:+.0%}; {gap / visit.loc['ATE', 'se']:+.1f} SE), and the "
        f"nonparametric DML adjustment agrees ({dml_pool['value'] * 100:.3f} pp). Treatment in the released "
        "file is slightly predictable from features that predict visits. The gate's tolerances, set before "
        "any outcome was read, did not anticipate that so small an imbalance could matter this much. In "
        f"scenario A the adjusted estimators' visit bias against the difference in means ranges from "
        f"{min(a_bias):+.0%} to {max(a_bias):+.0%}, the same order as this gap.",
        "- **Scenario A, against the adjusted reference:** "
        + (", ".join(recover_adjusted) if recover_adjusted else "no ATE estimator")
        + " recover it within Monte Carlo error on visit.",
        f"- **Nonlinear selection (N):** linear-propensity IPW is biased by {n_bias['ipw_ate']:+.0%} and "
        f"linear AIPW by {n_bias['aipw']:+.0%} against the difference in means "
        f"({n_adjusted['ipw_ate']:+.0%} and {n_adjusted['aipw']:+.0%} against the adjusted reference). "
        "DML with boosted nuisances is closer: "
        f"{n_adjusted['dml_irm']:+.0%} against the adjusted reference.",
        f"- **Hiding f6 (H):** the paired shift on visit is {moved['ipw_ate']:+.3f} pp for IPW, "
        f"{moved['aipw']:+.3f} pp for AIPW, {moved['outcome_regression']:+.3f} pp for outcome regression and "
        f"{moved['dml_irm']:+.3f} pp for DML. Hiding "
        "a selection feature moves each of these estimates; the size depends on the estimator and on how "
        "much of f6 the remaining features carry. This is a shift between two analyses of the same "
        "users, not by "
        "itself a bias against either reference.",
        f"- **Poor overlap (P):** the oracle's own visit interval covers its reference in only "
        f"{oracle_p_coverage:.0%} of draws, so interval coverage is not informative "
        "there; DoubleML's nu2 fallbacks are also concentrated in this scenario.",
        "- **Conversion** is rare (about 0.3%), so even the oracle's Monte Carlo error is large; conversion "
        "results support few conclusions either way.",
        "",
    ]
    return lines


def render_extension_report(paths: DataPaths | None = None):
    from causeway import visualization as viz
    from causeway.reporting import ReportOutputs

    paths = paths or DataPaths()
    protocol = load_protocol()
    results = paths.results("extension")
    tables = {name: pd.read_csv(results / f"{name}.csv") for name in EXTENSION_TABLES}
    if not all((results / f"{name}.csv").exists() for name in POST_HOC_TABLES):
        compute_post_hoc(paths)
    tables.update({name: pd.read_csv(results / f"{name}.csv") for name in POST_HOC_TABLES})
    provenance = json.loads((results / "provenance.json").read_text())
    provenance["post_hoc"] = json.loads((results / "post_hoc_provenance.json").read_text())
    outcomes = [protocol["outcomes"]["primary"], *protocol["outcomes"]["secondary"]]
    figure = viz.extension_heatmap(
        tables["summary"], paths.figures("extension") / "extension_recovery.png", FIGURE_ESTIMATORS, outcomes
    )
    report = results / "extension_report.md"
    report.write_text(extension_markdown(tables, protocol, provenance))
    return ReportOutputs(tables, [figure], report)


def run_extension_report(paths: DataPaths | None = None, n_jobs: int = 8):
    compute_extension(paths, n_jobs)
    return render_extension_report(paths)
