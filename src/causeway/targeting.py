"""Targeting policies from cross-fitted CATE predictions, scored by IPW on held-out customers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from causeway.heterogeneity import causal_forest

ARM_CODES = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
EMAIL_ARMS = {"mens": 1, "womens": 2}


@dataclass
class CrossFittedEffects:
    arm: np.ndarray  # randomized arm code per customer
    y: np.ndarray
    cate: dict[str, np.ndarray]  # out-of-fold predicted effect of each e-mail versus none


def cross_fitted_effects(
    frame: pd.DataFrame, x: np.ndarray, outcome: str, plan: dict, seed: int
) -> CrossFittedEffects:
    """Out-of-fold CATE predictions for both e-mails, for every customer in the trial."""
    arm = frame["segment"].astype(str).map(ARM_CODES).to_numpy()
    y = frame[outcome].to_numpy(dtype=float)
    cate = {name: np.empty(len(y)) for name in EMAIL_ARMS}
    folds = KFold(n_splits=plan["targeting"]["cross_fitting_folds"], shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(folds.split(x)):
        for name, code in EMAIL_ARMS.items():
            in_pair = train[np.isin(arm[train], [0, code])]
            forest = causal_forest(plan, seed + fold).fit(
                y[in_pair], (arm[in_pair] == code).astype(int), X=x[in_pair]
            )
            cate[name][test] = np.ravel(forest.effect(x[test]))
    return CrossFittedEffects(arm, y, cate)


def policy_value(
    assignment: np.ndarray, arm: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Hajek IPW value of a deterministic policy: customers whose randomized arm matches the assignment,
    each weighted by 1 / (share of their arm). `weights` allows bootstrap resampling counts."""
    shares = np.array([np.mean(arm == a) for a in range(3)])
    w = (arm == assignment) / shares[arm]
    if weights is not None:
        w = w * weights
    return float((w * y).sum() / w.sum())


def top_fraction(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Indicator of the top `fraction` of customers by score (ties broken by position, deterministic)."""
    n_selected = int(round(fraction * len(scores)))
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(-scores, kind="stable")[:n_selected]] = True
    return selected


def policies(effects: CrossFittedEffects, budget: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The pre-specified policies, as an arm code for every customer."""
    n = len(effects.y)
    m, w = effects.cate["mens"], effects.cate["womens"]
    best_arm = np.where(np.maximum(m, w) <= 0, 0, np.where(m >= w, 1, 2))
    best_gain = np.maximum(np.maximum(m, w), 0)
    random_email = rng.integers(1, 3, size=n)
    random_budget = top_fraction(rng.random(n), budget)
    return {
        "treat_none": np.zeros(n, dtype=int),
        "treat_all_mens": np.ones(n, dtype=int),
        "treat_all_womens": np.full(n, 2),
        "random_email": random_email,
        "model_best_arm": best_arm,
        "budget_random_mens": np.where(random_budget, 1, 0),
        "budget_model_mens": np.where(top_fraction(m, budget), 1, 0),
        "budget_model_womens": np.where(top_fraction(w, budget), 2, 0),
        "budget_model_best_arm": np.where(top_fraction(best_gain, budget) & (best_arm > 0), best_arm, 0),
    }


def evaluate_policies(
    effects: CrossFittedEffects,
    budget: float,
    n_bootstrap: int,
    seed: int,
    comparisons: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Policy values relative to treating nobody, with bootstrap CIs.

    Customers are resampled within arms and the predictions are held fixed.
    """
    rng = np.random.default_rng(seed)
    assigned = policies(effects, budget, rng)
    base = policy_value(assigned["treat_none"], effects.arm, effects.y)
    point = {name: policy_value(a, effects.arm, effects.y) - base for name, a in assigned.items()}
    draws = {name: np.empty(n_bootstrap) for name in assigned}
    arm_index = [np.flatnonzero(effects.arm == a) for a in range(3)]
    for b in range(n_bootstrap):
        counts = np.zeros(len(effects.y))
        for index in arm_index:
            np.add.at(counts, rng.choice(index, size=len(index)), 1.0)
        none_value = policy_value(assigned["treat_none"], effects.arm, effects.y, counts)
        for name, a in assigned.items():
            draws[name][b] = policy_value(a, effects.arm, effects.y, counts) - none_value
    values = pd.DataFrame(
        [
            {
                "policy": name,
                "share_emailed": float((a > 0).mean()),
                "incremental_value": point[name],
                "ci_low": float(np.percentile(draws[name], 2.5)),
                "ci_high": float(np.percentile(draws[name], 97.5)),
            }
            for name, a in assigned.items()
        ]
    )
    differences = pd.DataFrame(
        [
            {
                "policy": left,
                "versus": right,
                "difference": point[left] - point[right],
                "ci_low": float(np.percentile(draws[left] - draws[right], 2.5)),
                "ci_high": float(np.percentile(draws[left] - draws[right], 97.5)),
            }
            for left, right in comparisons
        ]
    )
    differences["interval_excludes_zero"] = (differences["ci_low"] > 0) | (differences["ci_high"] < 0)
    return values, differences


def uplift_curve(effects: CrossFittedEffects, email: str, grid: np.ndarray) -> pd.DataFrame:
    """Incremental value per customer when the top q fraction (by predicted effect) receive `email`."""
    code = EMAIL_ARMS[email]
    none = np.zeros(len(effects.y), dtype=int)
    base = policy_value(none, effects.arm, effects.y)
    everyone = policy_value(np.full(len(effects.y), code), effects.arm, effects.y) - base
    rows = []
    for q in grid:
        assignment = np.where(top_fraction(effects.cate[email], q), code, 0)
        rows.append(
            {
                "email": email,
                "fraction_targeted": float(q),
                "model_incremental_value": policy_value(assignment, effects.arm, effects.y) - base,
                "random_incremental_value": float(q) * everyone,
            }
        )
    return pd.DataFrame(rows)
