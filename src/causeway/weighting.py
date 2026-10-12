"""Inverse probability weighting for the ATE and the ATT, with weight diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Estimate:
    estimate: float
    se: float
    ci_low: float
    ci_high: float


def ate_weights(scores: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """1/e for treated units, 1/(1-e) for controls: reweights both arms to the full sample."""
    return np.where(treated, 1.0 / scores, 1.0 / (1.0 - scores))


def att_weights(scores: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """1 for treated units, e/(1-e) for controls: reweights the controls to resemble the treated."""
    return np.where(treated, 1.0, scores / (1.0 - scores))


def hajek_difference(
    y: np.ndarray, treated: np.ndarray, weights: np.ndarray, ci_level: float = 0.95
) -> Estimate:
    """Difference of Hajek-normalised weighted means.

    The SE treats the weights as fixed, which is conservative for the ATE when the propensity is estimated.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(treated, dtype=bool)
    means, variance = [], 0.0
    for mask in (t, ~t):
        w, values = weights[mask], y[mask]
        mean = (w * values).sum() / w.sum()
        variance += ((w * (values - mean)) ** 2).sum() / w.sum() ** 2
        means.append(mean)
    estimate, se = means[0] - means[1], float(np.sqrt(variance))
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    return Estimate(float(estimate), se, float(estimate - z * se), float(estimate + z * se))


def kish_ess(weights: np.ndarray) -> float:
    """Kish effective sample size, (sum w)^2 / sum w^2."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w**2).sum())


def weight_diagnostics(weights: np.ndarray, treated: np.ndarray, prefix: str) -> dict[str, float]:
    """Per-arm effective sample size and the distribution of weights normalized to mean 1 within arm."""
    out = {}
    for arm, mask in (("treated", treated), ("control", ~treated)):
        w = weights[mask]
        normalized = w / w.mean()
        out[f"{prefix}_ess_{arm}"] = kish_ess(w)
        out[f"{prefix}_ess_share_{arm}"] = kish_ess(w) / len(w)
        out[f"{prefix}_weight_min_{arm}"] = float(normalized.min())
        out[f"{prefix}_weight_max_{arm}"] = float(normalized.max())
        for q in (1, 50, 99):
            out[f"{prefix}_weight_p{q}_{arm}"] = float(np.percentile(normalized, q))
    return out


def trim_mask(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    """Units kept when trimming on the estimated propensity; trimming changes the estimand."""
    return (scores >= low) & (scores <= high)
