"""Double Machine Learning for the ATE (interactive regression model).

DML uses flexible learners for the nuisance functions mu_t(x) and e(x) but estimates the effect with the
orthogonal (AIPW) score and cross-fitting. Orthogonality makes the estimate insensitive, to first order,
to errors in the nuisance fits, so they can converge slowly (faster than n^(-1/4)) without biasing the
estimate. Cross-fitting removes the own-observation overfitting bias that would otherwise arise from using
the same data to fit a flexible learner and to evaluate the score. Simply fitting a powerful outcome
model and averaging its predictions has neither property.

The primary implementation is DoubleML's IRM; `hand_dml` reproduces it with the same learners and folds.
"""

from __future__ import annotations

import doubleml
import numpy as np
import pandas as pd
from doubleml.utils import PSProcessorConfig
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from causeway.estimators import CrossFitResult, Estimate, cross_fitted_scores

# The frozen specification applies no trimming or clipping in the primary analysis. DoubleML clips
# propensities to [0.01, 0.99] by default; this threshold disables that in practice while staying within
# the library's accepted range.
NO_CLIPPING = PSProcessorConfig(clipping_threshold=1e-12, extreme_threshold=1e-12)


def gbm_classifier(seed: int) -> HistGradientBoostingClassifier:
    """Default hyperparameters; the seed only fixes the early-stopping validation split."""
    return HistGradientBoostingClassifier(random_state=seed)


def gbm_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=seed)


def doubleml_irm(
    y: np.ndarray,
    treated: np.ndarray,
    covariates: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    ci_level: float = 0.95,
) -> tuple[Estimate, dict[str, np.ndarray]]:
    """ATE from doubleml.DoubleMLIRM with the given folds (n_rep = 1, unnormalized IPW part).

    Also returns the out-of-fold nuisance predictions (g0, g1, m) for predictive diagnostics."""
    data = covariates.reset_index(drop=True).assign(
        _y=np.asarray(y, dtype=float), _d=np.asarray(treated, int)
    )
    model = doubleml.DoubleMLIRM(
        doubleml.DoubleMLData(data, "_y", "_d", list(covariates.columns)),
        ml_g=gbm_regressor(seed),
        ml_m=gbm_classifier(seed),
        n_folds=len(folds),
        n_rep=1,
        score="ATE",
        ps_processor_config=NO_CLIPPING,
        draw_sample_splitting=False,
    )
    model.set_sample_splitting([[(train, test) for train, test in folds]])
    model.fit()
    low, high = model.confint(level=ci_level).to_numpy()[0]
    predictions = {name: model.predictions[name][:, 0, 0] for name in ("ml_g0", "ml_g1", "ml_m")}
    return Estimate(float(model.coef[0]), float(model.se[0]), float(low), float(high)), predictions


def hand_dml(
    y: np.ndarray,
    treated: np.ndarray,
    covariates: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> CrossFitResult:
    """The same orthogonal-score estimator written out directly, for validation against DoubleML."""
    return cross_fitted_scores(
        y, treated, covariates, folds, lambda: gbm_classifier(seed), lambda: gbm_regressor(seed)
    )
