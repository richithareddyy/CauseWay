"""Covariate balance diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardized_mean_difference(
    treated: np.ndarray | pd.Series,
    control: np.ndarray | pd.Series,
    treated_weights: np.ndarray | None = None,
    control_weights: np.ndarray | None = None,
) -> float:
    """Difference in (weighted) means divided by the pooled unweighted standard deviation.

    The denominator uses the unweighted sample variances (ddof=1) of the two groups, so the scale is
    fixed before any adjustment and weighted and unweighted SMDs are comparable. Returns 0 when both
    groups are constant and equal, and inf when they are constant but different.
    """
    t = np.asarray(treated, dtype=float)
    c = np.asarray(control, dtype=float)
    mean_t = np.average(t, weights=treated_weights)
    mean_c = np.average(c, weights=control_weights)
    pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    difference = mean_t - mean_c
    if pooled_sd == 0:
        return 0.0 if difference == 0 else float(np.copysign(np.inf, difference))
    return float(difference / pooled_sd)


def balance_table(frame: pd.DataFrame, treatment: pd.Series, covariates: list[str]) -> pd.DataFrame:
    """Treated mean, control mean and SMD for each numeric covariate column."""
    is_treated = treatment.astype(bool).to_numpy()
    rows = []
    for column in covariates:
        values = frame[column].to_numpy(dtype=float)
        rows.append(
            {
                "covariate": column,
                "treated_mean": values[is_treated].mean(),
                "control_mean": values[~is_treated].mean(),
                "smd": standardized_mean_difference(values[is_treated], values[~is_treated]),
            }
        )
    return pd.DataFrame(rows)
