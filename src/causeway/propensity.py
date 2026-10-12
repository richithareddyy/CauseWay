"""Propensity score model and its diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import special
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass
class PropensityFit:
    scores: np.ndarray
    coefficients: dict[str, float]
    converged: bool


def fit_logistic_propensity(covariates: pd.DataFrame, treated: np.ndarray) -> PropensityFit:
    """Unpenalized maximum-likelihood logistic regression of treatment on the covariates (with intercept)."""
    design = sm.add_constant(covariates.to_numpy(dtype=float), has_constant="add")
    fit = sm.Logit(np.asarray(treated, dtype=float), design).fit(disp=0, maxiter=200)
    names = ["intercept", *covariates.columns]
    scores = special.expit(design @ fit.params)
    return PropensityFit(
        scores,
        dict(zip(names, fit.params.tolist(), strict=True)),
        bool(fit.mle_retvals.get("converged", False)),
    )


def predictive_metrics(scores: np.ndarray, treated: np.ndarray) -> dict[str, float]:
    """In-sample ability to predict treatment. Not a measure of causal adequacy."""
    t = np.asarray(treated, dtype=int)
    clipped = np.clip(scores, 1e-12, 1 - 1e-12)
    return {
        "auc": float(roc_auc_score(t, scores)),
        "log_loss": float(log_loss(t, clipped)),
        "brier": float(brier_score_loss(t, scores)),
    }


def overlap_metrics(scores: np.ndarray, treated: np.ndarray, bounds: list[list[float]]) -> dict[str, float]:
    """Positivity diagnostics: extremes and the share of units outside each trimming band."""
    t = np.asarray(treated, dtype=bool)
    out = {
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "treated_score_min": float(scores[t].min()),
        "control_score_max": float(scores[~t].max()),
    }
    for low, high in bounds:
        out[f"share_outside_{low:.2f}_{high:.2f}"] = float(((scores < low) | (scores > high)).mean())
    return out


def recovery_metrics(scores: np.ndarray, e_star: np.ndarray) -> dict[str, float]:
    """Distance between the estimated and the true design propensity (available only in the benchmark)."""
    logit_gap = special.logit(np.clip(scores, 1e-12, 1 - 1e-12)) - special.logit(e_star)
    return {
        "rmse_vs_true_propensity": float(np.sqrt(np.mean((scores - e_star) ** 2))),
        "max_abs_error_vs_true_propensity": float(np.max(np.abs(scores - e_star))),
        "rmse_logit_vs_true": float(np.sqrt(np.mean(logit_gap**2))),
    }
