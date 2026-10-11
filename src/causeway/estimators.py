"""Outcome regression (g-computation) and cross-fitted augmented IPW.

The AIPW score combines an outcome model mu_t(x) = E[Y | T=t, X=x] with a propensity model e(x):

    psi_i = mu_1(x_i) - mu_0(x_i) + T_i (Y_i - mu_1(x_i)) / e(x_i) - (1 - T_i)(Y_i - mu_0(x_i)) / (1 - e(x_i))

Its mean is consistent for the ATE if either model is correct (double robustness), and its first-order
sensitivity to errors in either model vanishes (Neyman orthogonality). Nuisance models are fitted on K-1
folds and evaluated on the held-out fold (cross-fitting), so no unit's own outcome informs its
prediction. The same function serves the linear AIPW estimator and, with flexible learners, the
hand-written check on Double Machine Learning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causeway.experiment import lin_adjusted


def cross_fit_folds(n: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled K-fold splits shared by every cross-fitted estimator within one draw."""
    return list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.arange(n)))


def linear_propensity_model() -> ClassifierMixin:
    """Unpenalized logistic regression (C = inf); standardization only aids convergence."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000))


def linear_outcome_model() -> RegressorMixin:
    return LinearRegression()


# --------------------------------------------------------------------------------------------------
# Outcome regression (g-computation)
# --------------------------------------------------------------------------------------------------


@dataclass
class Estimate:
    estimate: float
    se: float
    ci_low: float
    ci_high: float


def g_computation(y: np.ndarray, treated: np.ndarray, covariates: pd.DataFrame) -> float:
    """Average over the sample of mu_1(x) - mu_0(x), with OLS fitted separately in each arm."""
    x = covariates.to_numpy(dtype=float)
    mu1 = LinearRegression().fit(x[treated], y[treated]).predict(x)
    mu0 = LinearRegression().fit(x[~treated], y[~treated]).predict(x)
    return float((mu1 - mu0).mean())


def outcome_regression(
    y: np.ndarray, treated: np.ndarray, covariates: pd.DataFrame, ci_level: float = 0.95
) -> Estimate:
    """G-computation with arm-specific OLS. Its point estimate equals the treatment coefficient of the
    centred, fully interacted regression, whose HC2 standard error is reported (frozen v1 rule)."""
    lin = lin_adjusted(y, treated, covariates, ci_level)
    return Estimate(lin.estimate, lin.se, lin.ci_low, lin.ci_high)


# --------------------------------------------------------------------------------------------------
# Cross-fitted AIPW
# --------------------------------------------------------------------------------------------------


@dataclass
class CrossFitResult:
    psi: np.ndarray  # (n, k) orthogonal scores, one column per outcome
    propensity: np.ndarray  # out-of-fold e(x)
    mu1: np.ndarray  # (n, k) out-of-fold predictions
    mu0: np.ndarray

    def estimates(self, ci_level: float = 0.95) -> list[Estimate]:
        n = self.psi.shape[0]
        z = stats.norm.ppf(1 - (1 - ci_level) / 2)
        out = []
        for column in self.psi.T:
            estimate, se = float(column.mean()), float(column.std(ddof=1) / np.sqrt(n))
            out.append(Estimate(estimate, se, estimate - z * se, estimate + z * se))
        return out


def cross_fitted_scores(
    y: np.ndarray,
    treated: np.ndarray,
    covariates: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    propensity_model: Callable[[], ClassifierMixin],
    outcome_model: Callable[[], RegressorMixin],
) -> CrossFitResult:
    """Out-of-fold nuisance predictions and AIPW scores for one or more outcomes (no clipping)."""
    x = covariates.to_numpy(dtype=float)
    y2 = np.asarray(y, dtype=float).reshape(len(y), -1)
    t = np.asarray(treated, dtype=bool)
    propensity = np.empty(len(t))
    mu1, mu0 = np.empty_like(y2), np.empty_like(y2)
    for train, test in folds:
        propensity[test] = propensity_model().fit(x[train], t[train]).predict_proba(x[test])[:, 1]
        treated_train, control_train = train[t[train]], train[~t[train]]
        for k in range(y2.shape[1]):
            mu1[test, k] = outcome_model().fit(x[treated_train], y2[treated_train, k]).predict(x[test])
            mu0[test, k] = outcome_model().fit(x[control_train], y2[control_train, k]).predict(x[test])
    e = propensity[:, None]
    tt = t[:, None]
    psi = mu1 - mu0 + tt * (y2 - mu1) / e - (~tt) * (y2 - mu0) / (1 - e)
    return CrossFitResult(psi, propensity, mu1, mu0)


class ArmwiseRegression(BaseEstimator, RegressorMixin):
    """Fit a separate regression in each arm, with the treatment indicator as the LAST feature column.

    Lets EconML's DR learner (which regresses Y on [X, T]) use exactly the arm-specific outcome models
    of the hand-written AIPW, so the two implementations can be compared like for like.
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, features, y):
        features = np.asarray(features, dtype=float)
        x, t = features[:, :-1], features[:, -1].astype(bool)
        base = self.base if self.base is not None else LinearRegression()
        self.model_1_ = clone(base).fit(x[t], y[t])
        self.model_0_ = clone(base).fit(x[~t], y[~t])
        return self

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        x, t = features[:, :-1], features[:, -1].astype(bool)
        return np.where(t, self.model_1_.predict(x), self.model_0_.predict(x))
