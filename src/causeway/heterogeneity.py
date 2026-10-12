"""Heterogeneous effects in the randomized experiment: BLP/GATES/CLAN, segments and a causal forest."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import CausalForestDML
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.stats.multitest import multipletests

from causeway.config import CONFIG_DIR
from causeway.experiment import neyman_difference

Z_POWER = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)


def load_heterogeneity_plan(path: Path = CONFIG_DIR / "heterogeneity.toml") -> dict:
    return tomllib.loads(path.read_text())


def plan_seed(base_seed: int, stream: int, *key: int) -> int:
    return int(np.random.SeedSequence(entropy=base_seed, spawn_key=(stream, *key)).generate_state(1)[0])


def causal_forest(plan: dict, seed: int) -> CausalForestDML:
    settings = plan["causal_forest"]
    return CausalForestDML(
        model_y=HistGradientBoostingRegressor(random_state=seed),
        model_t=DummyClassifier(strategy="prior"),
        discrete_treatment=True,
        n_estimators=settings["n_estimators"],
        min_samples_leaf=settings["min_samples_leaf"],
        honest=settings["honest"],
        cv=settings["cross_fitting_folds"],
        random_state=seed,
        n_jobs=1,
    )


def _ols(y: np.ndarray, design: np.ndarray):
    return sm.OLS(y, design).fit(cov_type="HC1")


def blp_gates_clan(
    y: np.ndarray,
    treated: np.ndarray,
    x: np.ndarray,
    covariate_names: list[str],
    proxy_cate: np.ndarray,
    proxy_baseline: np.ndarray,
    n_groups: int,
    ci_level: float,
) -> dict:
    """BLP, GATES and CLAN on the main half, given proxies fitted on the auxiliary half."""
    t = treated.astype(float)
    p = t.mean()
    s_centred = proxy_cate - proxy_cate.mean()
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    out = {}

    blp = _ols(y, np.column_stack([np.ones_like(t), proxy_baseline, t - p, (t - p) * s_centred]))
    for name, index in (("b1", 2), ("b2", 3)):
        estimate, se = float(blp.params[index]), float(blp.bse[index])
        out[name] = {
            "estimate": estimate,
            "se": se,
            "ci_low": estimate - z * se,
            "ci_high": estimate + z * se,
            "p_value": float(blp.pvalues[index]),
        }

    ranks = stats.rankdata(proxy_cate, method="ordinal")
    groups = np.minimum((ranks - 1) * n_groups // len(ranks), n_groups - 1)
    dummies = np.column_stack([(groups == g).astype(float) for g in range(n_groups)])
    gates = _ols(y, np.column_stack([np.ones_like(t), proxy_baseline, (t - p)[:, None] * dummies]))
    for g in range(n_groups):
        estimate, se = float(gates.params[2 + g]), float(gates.bse[2 + g])
        out[f"gate_{g + 1}"] = {
            "estimate": estimate,
            "se": se,
            "ci_low": estimate - z * se,
            "ci_high": estimate + z * se,
            "p_value": float(gates.pvalues[2 + g]),
        }
    contrast = np.zeros(len(gates.params))
    contrast[2 + n_groups - 1], contrast[2] = 1.0, -1.0
    test = gates.t_test(contrast)
    estimate, se = float(np.squeeze(test.effect)), float(np.squeeze(test.sd))
    out["gate_top_minus_bottom"] = {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "p_value": float(np.squeeze(test.pvalue)),
    }

    top, bottom = groups == n_groups - 1, groups == 0
    for j, name in enumerate(covariate_names):
        a, b = x[top, j], x[bottom, j]
        estimate = float(a.mean() - b.mean())
        se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        p_value = float(2 * stats.norm.sf(abs(estimate) / se)) if se > 0 else float(estimate == 0)
        out[f"clan_{name}"] = {
            "estimate": estimate,
            "se": se,
            "ci_low": estimate - z * se,
            "ci_high": estimate + z * se,
            "p_value": p_value,
            "top_mean": float(a.mean()),
            "bottom_mean": float(b.mean()),
        }
    return out


def run_split(
    y: np.ndarray, treated: np.ndarray, x: np.ndarray, covariate_names: list[str], plan: dict, seed: int
) -> dict:
    """One repetition: random auxiliary/main split, proxies fitted on the auxiliary half."""
    rng = np.random.default_rng(seed)
    auxiliary = rng.random(len(y)) < plan["blp"]["auxiliary_share"]
    main = ~auxiliary
    forest = causal_forest(plan, seed).fit(y[auxiliary], treated[auxiliary].astype(int), X=x[auxiliary])
    proxy_cate = np.ravel(forest.effect(x[main]))
    controls = auxiliary & ~treated
    baseline = HistGradientBoostingRegressor(random_state=seed).fit(x[controls], y[controls]).predict(x[main])
    alpha = plan["blp"]["alpha"]
    # Per-split intervals at level 1 - alpha/2; their median bounds have coverage of at least 1 - alpha.
    return blp_gates_clan(
        y[main],
        treated[main],
        x[main],
        covariate_names,
        proxy_cate,
        baseline,
        plan["blp"]["gates_groups"],
        1 - alpha / 2,
    )


def aggregate_splits(splits: list[dict]) -> pd.DataFrame:
    """Combine splits by medians, with p = 2 x median p (capped at 1).

    share_positive flags estimates whose sign changes from split to split.
    """
    rows = []
    for parameter in splits[0]:
        values = pd.DataFrame([s[parameter] for s in splits])
        row = {
            "parameter": parameter,
            "estimate": values["estimate"].median(),
            "ci_low": values["ci_low"].median(),
            "ci_high": values["ci_high"].median(),
            "p_value": min(1.0, 2 * values["p_value"].median()),
            "share_positive": float((values["estimate"] > 0).mean()),
            "n_splits": len(values),
        }
        for extra in ("top_mean", "bottom_mean"):
            if extra in values:
                row[extra] = values[extra].median()
        rows.append(row)
    return pd.DataFrame(rows)


def heterogeneity_verdict(aggregated: pd.DataFrame, alpha: float) -> dict:
    """Pre-specified rule: b2 > 0 with adjusted p < alpha AND the top-minus-bottom GATES interval
    excludes 0."""
    table = aggregated.set_index("parameter")
    b2, spread = table.loc["b2"], table.loc["gate_top_minus_bottom"]
    blp_evidence = bool(b2["estimate"] > 0 and b2["p_value"] < alpha)
    gates_evidence = bool(spread["ci_low"] > 0 or spread["ci_high"] < 0)
    return {
        "b2": b2["estimate"],
        "b2_p_value": b2["p_value"],
        "gates_spread": spread["estimate"],
        "gates_spread_ci_low": spread["ci_low"],
        "gates_spread_ci_high": spread["ci_high"],
        "blp_evidence": blp_evidence,
        "gates_evidence": gates_evidence,
        "heterogeneity_detected": blp_evidence and gates_evidence,
    }


def segment_effects(y: np.ndarray, treated: np.ndarray, levels: pd.Series, variable: str) -> pd.DataFrame:
    """Difference in means within each level of one pre-specified variable, with its minimum
    detectable effect."""
    rows = []
    for level in sorted(levels.astype(str).unique()):
        in_level = (levels.astype(str) == level).to_numpy()
        a, b = y[in_level & treated], y[in_level & ~treated]
        effect = neyman_difference(a, b)
        rows.append(
            {
                "variable": variable,
                "level": level,
                "n_treated": len(a),
                "n_control": len(b),
                "estimate": effect.estimate,
                "se": effect.se,
                "ci_low": effect.ci_low,
                "ci_high": effect.ci_high,
                "p_value": effect.p_value,
                "mde": Z_POWER * effect.se,
            }
        )
    return pd.DataFrame(rows)


def equality_test(levels: pd.DataFrame) -> dict:
    """Wald chi-square test that all levels share one effect (precision-weighted)."""
    usable = levels[levels["se"] > 0]
    weights = 1 / usable["se"] ** 2
    pooled = float((weights * usable["estimate"]).sum() / weights.sum())
    statistic = float((weights * (usable["estimate"] - pooled) ** 2).sum())
    dof = len(usable) - 1
    return {
        "chi2": statistic,
        "dof": dof,
        "p_value": float(stats.chi2.sf(statistic, dof)) if dof > 0 else np.nan,
    }


def classify(estimate: float, q_value: float, alpha: float = 0.05) -> str:
    if q_value < alpha:
        return "benefit" if estimate > 0 else "harm"
    return "inconclusive"


def segment_analysis(frame: pd.DataFrame, treated: np.ndarray, outcome: str, variables: list[str]) -> tuple:
    y = frame[outcome].to_numpy(dtype=float)
    levels = pd.concat([segment_effects(y, treated, frame[v], v) for v in variables], ignore_index=True)
    levels["q_value"] = multipletests(levels["p_value"], method="fdr_bh")[1]
    levels["classification"] = [
        classify(e, q) for e, q in zip(levels["estimate"], levels["q_value"], strict=True)
    ]
    tests = pd.DataFrame(
        [{"variable": v, **equality_test(levels[levels["variable"] == v])} for v in variables]
    )
    tests["q_value"] = multipletests(tests["p_value"], method="fdr_bh")[1]
    return levels, tests


@dataclass
class ForestDescription:
    cate: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    importances: dict[str, float]


def describe_with_forest(
    y: np.ndarray, treated: np.ndarray, x: np.ndarray, names: list[str], plan: dict, seed: int
) -> ForestDescription:
    """Causal forest on the whole pair: predicted effects with pointwise intervals (descriptive only)."""
    forest = causal_forest(plan, seed).fit(y, treated.astype(int), X=x)
    low, high = forest.effect_interval(x, alpha=0.05)
    return ForestDescription(
        np.ravel(forest.effect(x)),
        np.ravel(low),
        np.ravel(high),
        dict(zip(names, forest.feature_importances_.tolist(), strict=True)),
    )


def split_task(
    y: np.ndarray, treated: np.ndarray, x: np.ndarray, names: list[str], plan: dict, seed: int
) -> dict:
    """One split with a single thread, for parallel execution."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return run_split(y, treated, x, names, plan, seed)
