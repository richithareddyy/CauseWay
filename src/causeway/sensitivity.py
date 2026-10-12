"""Sensitivity to unobserved confounding: omitted-variable bounds, E-values and Rosenbaum bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats


def partial_r2(t_value: float, dof: float) -> float:
    return t_value**2 / (t_value**2 + dof)


def robustness_value(t_value: float, dof: float, q: float = 1.0, alpha: float | None = None) -> float:
    """Cinelli & Hazlett (2020) robustness value: the partial R2 a confounder needs with both treatment and
    outcome to remove a share q of the estimate (or, with alpha, to make it insignificant)."""
    f = q * abs(t_value) / np.sqrt(dof)
    if alpha is not None:
        f -= stats.t.ppf(1 - alpha / 2, dof - 1) / np.sqrt(dof - 1)
        if f <= 0:
            return 0.0
    return float(0.5 * (np.sqrt(f**4 + 4 * f**2) - f**2))


def bias_bound(se: float, dof: float, r2_yz: float, r2_dz: float) -> float:
    """Largest absolute bias from a confounder with the given partial R2s with outcome and treatment."""
    return float(se * np.sqrt(dof) * np.sqrt(r2_yz * r2_dz / (1 - r2_dz)))


def benchmark_bounds(r2_dxj: float, r2_yxj: float, k: float) -> tuple[float, float]:
    """Partial R2s of a confounder k times as strong as an observed covariate X_j (Cinelli & Hazlett, 2020,
    Section 6.2, with k_d = k_y = k)."""
    r2_dz = k * r2_dxj / (1 - r2_dxj)
    if r2_dz >= 1:
        return np.nan, np.nan
    r2_zxj = k * r2_dxj**2 / ((1 - k * r2_dxj) * (1 - r2_dxj))
    r2_yz = ((np.sqrt(k) + np.sqrt(r2_zxj)) / np.sqrt(1 - r2_zxj)) ** 2 * (r2_yxj / (1 - r2_yxj))
    return float(min(r2_yz, 1.0)), float(r2_dz)


def ovb_analysis(
    y: np.ndarray,
    treated: np.ndarray,
    covariates: pd.DataFrame,
    multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> tuple[dict, pd.DataFrame]:
    """Linear outcome model Y ~ D + X: robustness values and covariate-benchmarked bounds."""
    x = covariates.to_numpy(dtype=float)
    d = np.asarray(treated, dtype=float)
    outcome = sm.OLS(y, sm.add_constant(np.column_stack([d, x]))).fit()
    treatment = sm.OLS(d, sm.add_constant(x)).fit()
    estimate, se, dof = float(outcome.params[1]), float(outcome.bse[1]), float(outcome.df_resid)
    t_value = estimate / se
    summary = {
        "estimate": estimate,
        "se": se,
        "dof": dof,
        "r2_yd_x": partial_r2(t_value, dof),
        "rv_q1": robustness_value(t_value, dof),
        "rv_q1_alpha05": robustness_value(t_value, dof, alpha=0.05),
    }
    rows = []
    for j, name in enumerate(covariates.columns):
        r2_dxj = partial_r2(treatment.tvalues[1 + j], treatment.df_resid)
        r2_yxj = partial_r2(outcome.tvalues[2 + j], dof)
        for k in multipliers:
            r2_yz, r2_dz = benchmark_bounds(r2_dxj, r2_yxj, k)
            bias = bias_bound(se, dof, r2_yz, r2_dz) if np.isfinite(r2_dz) else np.nan
            rows.append(
                {
                    "benchmark": name,
                    "k": k,
                    "r2_yz": r2_yz,
                    "r2_dz": r2_dz,
                    "bias_bound": bias,
                    "adjusted_estimate": estimate - np.sign(estimate) * bias,
                }
            )
    return summary, pd.DataFrame(rows)


def confounder_strength(
    y: np.ndarray, treated: np.ndarray, covariates: pd.DataFrame, confounder: np.ndarray
) -> dict:
    """Actual partial R2s of a variable with outcome and treatment, given the covariates."""
    x = covariates.to_numpy(dtype=float)
    d = np.asarray(treated, dtype=float)
    z = np.asarray(confounder, dtype=float)
    outcome = sm.OLS(y, sm.add_constant(np.column_stack([d, x, z]))).fit()
    treatment = sm.OLS(d, sm.add_constant(np.column_stack([x, z]))).fit()
    return {
        "r2_yz": partial_r2(outcome.tvalues[-1], outcome.df_resid),
        "r2_dz": partial_r2(treatment.tvalues[-1], treatment.df_resid),
    }


def e_value(risk_ratio: float) -> float:
    """VanderWeele & Ding (2017). Ratios below 1 are inverted first; a ratio of 1 gives 1."""
    if not np.isfinite(risk_ratio) or risk_ratio <= 0:
        return np.nan
    rr = risk_ratio if risk_ratio >= 1 else 1 / risk_ratio
    return float(rr + np.sqrt(rr * (rr - 1)))


def weighted_risk_ratio(
    y: np.ndarray, treated: np.ndarray, weights: np.ndarray, ci_level: float = 0.95
) -> dict:
    """Ratio of Hajek-weighted arm means with a delta-method interval on the log scale."""
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    parts = []
    for mask in (treated, ~treated):
        w, v = weights[mask], y[mask]
        mean = (w * v).sum() / w.sum()
        var = ((w * (v - mean)) ** 2).sum() / w.sum() ** 2
        parts.append((mean, var))
    (m1, v1), (m0, v0) = parts
    log_rr, se = np.log(m1 / m0), np.sqrt(v1 / m1**2 + v0 / m0**2)
    low = float(np.exp(log_rr - z * se))
    return {
        "risk_ratio": float(m1 / m0),
        "ci_low": low,
        "ci_high": float(np.exp(log_rr + z * se)),
        "e_value": e_value(m1 / m0),
        "e_value_ci": e_value(low) if low > 1 else 1.0,
    }


def rosenbaum_gamma(treated_only: int, control_only: int, alpha: float = 0.05) -> float:
    """Largest Gamma at which a one-sided McNemar test on discordant matched pairs still rejects.

    treated_only counts pairs where only the treated unit had the event; control_only the reverse. NaN means
    the test does not reject even at Gamma = 1.
    """
    n = treated_only + control_only
    if n == 0:
        return np.nan

    def p_upper(gamma: float) -> float:
        return float(stats.binom.sf(treated_only - 1, n, gamma / (1 + gamma)))

    if p_upper(1.0) > alpha:
        return np.nan
    high = 2.0
    while p_upper(high) <= alpha and high < 1e6:
        high *= 2
    return float(optimize.brentq(lambda g: p_upper(g) - alpha, 1.0, high))
