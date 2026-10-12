"""Design-based analysis of the randomized experiment (Part I)."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep

from causeway.config import CONFIG_DIR
from causeway.features import covariate_matrix

ESTIMAND = "ITT effect of assignment"


def load_experiment_plan(path: Path = CONFIG_DIR / "experiment.toml") -> dict:
    return tomllib.loads(path.read_text())


def stream_rng(base_seed: int, stream: int) -> np.random.Generator:
    """Independent, reproducible generator for one named stream of the analysis plan."""
    return np.random.default_rng(np.random.SeedSequence(entropy=base_seed, spawn_key=(stream,)))


def stream_int_seed(base_seed: int, stream: int) -> int:
    return int(np.random.SeedSequence(entropy=base_seed, spawn_key=(stream,)).generate_state(1)[0])


@dataclass
class Estimate:
    estimate: float
    se: float | None
    ci_low: float
    ci_high: float
    p_value: float | None = None


def neyman_difference(y_treated: np.ndarray, y_control: np.ndarray, ci_level: float = 0.95) -> Estimate:
    """Difference in means with the Neyman variance (the same as HC2 here)."""
    y_t = np.asarray(y_treated, dtype=float)
    y_c = np.asarray(y_control, dtype=float)
    estimate = y_t.mean() - y_c.mean()
    se = float(np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c)))
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    p_value = float(2 * stats.norm.sf(abs(estimate) / se)) if se > 0 else float(estimate == 0)
    return Estimate(float(estimate), se, estimate - z * se, estimate + z * se, p_value)


def lin_adjusted(
    y: np.ndarray, treated: np.ndarray, covariates: pd.DataFrame, ci_level: float = 0.95
) -> Estimate:
    """Lin (2013) regression adjustment: treatment interacted with centred covariates, HC2 SEs."""
    t = np.asarray(treated, dtype=float)
    centred = covariates.to_numpy(dtype=float) - covariates.to_numpy(dtype=float).mean(axis=0)
    design = np.column_stack([np.ones_like(t), t, centred, t[:, None] * centred])
    fit = sm.OLS(np.asarray(y, dtype=float), design).fit(cov_type="HC2")
    low, high = fit.conf_int(alpha=1 - ci_level)[1]
    return Estimate(float(fit.params[1]), float(fit.bse[1]), float(low), float(high), float(fit.pvalues[1]))


def randomization_test(
    y_treated: np.ndarray,
    y_control: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    batch_size: int = 100,
) -> np.ndarray:
    """Two-sided Fisher randomization test of the sharp null.

    p = (1 + #exceed) / (1 + R); several outcomes can be tested at once.
    """
    y_t = np.asarray(y_treated, dtype=float).reshape(len(y_treated), -1)
    y_c = np.asarray(y_control, dtype=float).reshape(len(y_control), -1)
    pooled = np.vstack([y_t, y_c])
    n, n_t = len(pooled), len(y_t)
    total = pooled.sum(axis=0)
    observed = np.abs(y_t.mean(axis=0) - y_c.mean(axis=0))
    threshold = observed - 1e-12 * np.maximum(1.0, observed)
    exceed = np.zeros(pooled.shape[1])
    done = 0
    while done < n_permutations:
        size = min(batch_size, n_permutations - done)
        treated_idx = rng.random((size, n)).argpartition(n_t, axis=1)[:, :n_t]
        sum_t = pooled[treated_idx].sum(axis=1)
        permuted = np.abs(sum_t / n_t - (total - sum_t) / (n - n_t))
        exceed += (permuted >= threshold).sum(axis=0)
        done += size
    return (1 + exceed) / (1 + n_permutations)


def bootstrap_means(
    samples: list[np.ndarray], n_resamples: int, rng: np.random.Generator, batch_size: int = 100
) -> list[np.ndarray]:
    """Resample customers within each arm and return a (B, k) array of means per arm."""
    out = []
    for sample in samples:
        values = np.asarray(sample, dtype=float).reshape(len(sample), -1)
        n = len(values)
        means = np.empty((n_resamples, values.shape[1]))
        for start in range(0, n_resamples, batch_size):
            size = min(batch_size, n_resamples - start)
            means[start : start + size] = values[rng.integers(0, n, size=(size, n))].mean(axis=1)
        out.append(means)
    return out


def dim_acceleration(y_treated: np.ndarray, y_control: np.ndarray) -> float:
    """BCa acceleration for a difference in means.

    The two-sample jackknife has a closed form here; it matches scipy.stats.bootstrap.
    """
    y_t = np.asarray(y_treated, dtype=float)
    y_c = np.asarray(y_control, dtype=float)
    u_t = y_t - y_t.mean()
    u_c = -(y_c - y_c.mean())
    numerator = (u_t**3).sum() / len(y_t) ** 3 + (u_c**3).sum() / len(y_c) ** 3
    denominator = (u_t**2).sum() / len(y_t) ** 2 + (u_c**2).sum() / len(y_c) ** 2
    return float(numerator / (6 * denominator**1.5))


def percentile_interval(draws: np.ndarray, ci_level: float = 0.95) -> tuple[float, float]:
    alpha = (1 - ci_level) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return float(low), float(high)


def bca_interval(
    draws: np.ndarray, estimate: float, acceleration: float, ci_level: float = 0.95
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval (Efron, 1987)."""
    draws = np.asarray(draws, dtype=float)
    share_below = ((draws < estimate).sum() + (draws <= estimate).sum()) / (2 * len(draws))
    z0 = stats.norm.ppf(share_below)
    alpha = (1 - ci_level) / 2
    levels = []
    for z_alpha in (stats.norm.ppf(alpha), stats.norm.ppf(1 - alpha)):
        shifted = z0 + z_alpha
        levels.append(stats.norm.cdf(z0 + shifted / (1 - acceleration * shifted)))
    low, high = np.quantile(draws, levels)
    return float(low), float(high)


def binary_contrast(y_treated: np.ndarray, y_control: np.ndarray, ci_level: float = 0.95) -> dict[str, float]:
    """Risk difference (Newcombe CI) and risk ratio (log CI).

    The ratio is NaN when an arm has no events.
    """
    y_t = np.asarray(y_treated, dtype=int)
    y_c = np.asarray(y_control, dtype=int)
    k_t, n_t, k_c, n_c = int(y_t.sum()), len(y_t), int(y_c.sum()), len(y_c)
    alpha = 1 - ci_level
    rd_low, rd_high = confint_proportions_2indep(
        k_t, n_t, k_c, n_c, method="newcomb", compare="diff", alpha=alpha
    )
    if k_t > 0 and k_c > 0:
        risk_ratio = (k_t / n_t) / (k_c / n_c)
        rr_low, rr_high = confint_proportions_2indep(
            k_t, n_t, k_c, n_c, method="log", compare="ratio", alpha=alpha
        )
    else:
        risk_ratio = rr_low = rr_high = np.nan
    return {
        "events_treated": k_t,
        "n_treated": n_t,
        "events_control": k_c,
        "n_control": n_c,
        "risk_treated": k_t / n_t,
        "risk_control": k_c / n_c,
        "risk_difference": k_t / n_t - k_c / n_c,
        "rd_ci_low": float(rd_low),
        "rd_ci_high": float(rd_high),
        "risk_ratio": float(risk_ratio),
        "rr_ci_low": float(rr_low),
        "rr_ci_high": float(rr_high),
    }


def holm(p_values: list[float]) -> list[float]:
    return multipletests(p_values, method="holm")[1].tolist()


def assignment_diagnostics(covariates: pd.DataFrame, treated: np.ndarray, folds: int, seed: int) -> dict:
    """Can the covariates predict assignment? Under randomization they shouldn't."""
    t = np.asarray(treated, dtype=int)
    logit = sm.Logit(t, sm.add_constant(covariates.to_numpy(dtype=float))).fit(disp=0)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_predict(model, covariates.to_numpy(dtype=float), t, cv=cv, method="predict_proba")[
        :, 1
    ]
    return {
        "lr_statistic": float(logit.llr),
        "lr_df": int(logit.df_model),
        "lr_p_value": float(logit.llr_pvalue),
        "out_of_fold_auc": float(roc_auc_score(t, scores)),
    }


def assignment_by_strata(frame: pd.DataFrame, strata: list[str], arm_column: str = "segment") -> pd.DataFrame:
    """Arm shares within each level of each pre-treatment stratum, with a chi-square independence test."""
    rows = []
    for stratum in strata:
        table = pd.crosstab(frame[stratum].astype(str), frame[arm_column].astype(str))
        chi2, p_value, dof, _ = stats.chi2_contingency(table.to_numpy())
        shares = table.div(table.sum(axis=1), axis=0)
        for level, counts in table.iterrows():
            row = {"stratum": stratum, "level": level, "n": int(counts.sum())}
            row.update({f"share_{arm}": float(shares.loc[level, arm]) for arm in table.columns})
            row.update({"chi2": float(chi2), "dof": int(dof), "p_value": float(p_value)})
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class ExperimentResults:
    effects: pd.DataFrame
    binary: pd.DataFrame
    relative_lift: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    diagnostics: pd.DataFrame
    plan: dict = field(repr=False)

    def summary(self) -> dict:
        return {
            "effects": self.effects.to_dict(orient="records"),
            "binary": self.binary.to_dict(orient="records"),
            "relative_lift": self.relative_lift.to_dict(orient="records"),
            "diagnostics": self.diagnostics.to_dict(orient="records"),
        }


def _arm_values(frame: pd.DataFrame, arm: str, columns: list[str]) -> np.ndarray:
    return frame.loc[frame["segment"] == arm, columns].to_numpy(dtype=float)


def analyze_experiment(frame: pd.DataFrame, plan: dict | None = None) -> ExperimentResults:
    """Run every pre-specified Part I estimator on the processed dataset."""
    plan = plan or load_experiment_plan()
    seed = plan["randomness"]["base_seed"]
    ci_level = plan["inference"]["ci_level"]
    outcomes = [plan["outcomes"]["primary"], *plan["outcomes"]["secondary"]]
    contrasts = plan["contrasts"]
    arms = list(dict.fromkeys(arm for c in contrasts for arm in (c["control_arm"], c["treatment_arm"])))
    values = {arm: _arm_values(frame, arm, outcomes) for arm in arms}
    covariates = covariate_matrix(frame)[plan["regression_adjustment"]["covariates"]]

    boot_rng = stream_rng(seed, plan["randomness"]["bootstrap_stream"])
    boot = dict(
        zip(
            arms,
            bootstrap_means([values[a] for a in arms], plan["inference"]["bootstrap_resamples"], boot_rng),
            strict=True,
        )
    )
    perm_rng = stream_rng(seed, plan["randomness"]["permutation_stream"])

    effect_rows, binary_rows, lift_rows, draw_columns, diagnostic_rows = [], [], [], {}, []
    for contrast in contrasts:
        t_arm, c_arm = contrast["treatment_arm"], contrast["control_arm"]
        y_t, y_c = values[t_arm], values[c_arm]
        p_randomization = randomization_test(y_t, y_c, plan["inference"]["permutations"], perm_rng)
        pair = frame["segment"].isin([t_arm, c_arm]).to_numpy()
        treated_in_pair = (frame.loc[pair, "segment"] == t_arm).to_numpy()
        base = {
            "contrast": contrast["name"],
            "role": contrast["role"],
            "treatment_arm": t_arm,
            "control_arm": c_arm,
            "estimand": ESTIMAND,
            "n_treated": len(y_t),
            "n_control": len(y_c),
        }

        for k, outcome in enumerate(outcomes):
            neyman = neyman_difference(y_t[:, k], y_c[:, k], ci_level)
            lin = lin_adjusted(
                frame.loc[pair, outcome].to_numpy(), treated_in_pair, covariates.loc[pair], ci_level
            )
            draws = boot[t_arm][:, k] - boot[c_arm][:, k]
            draw_columns[f"{contrast['name']}__{outcome}"] = draws
            pct = percentile_interval(draws, ci_level)
            bca = bca_interval(draws, neyman.estimate, dim_acceleration(y_t[:, k], y_c[:, k]), ci_level)
            common = {**base, "outcome": outcome}
            effect_rows += [
                {
                    **common,
                    "estimator": "difference_in_means",
                    **asdict(neyman),
                    "p_randomization": float(p_randomization[k]),
                },
                {**common, "estimator": "lin_adjusted", **asdict(lin)},
                {
                    **common,
                    "estimator": "bootstrap_percentile",
                    "estimate": neyman.estimate,
                    "se": float(draws.std(ddof=1)),
                    "ci_low": pct[0],
                    "ci_high": pct[1],
                },
                {
                    **common,
                    "estimator": "bootstrap_bca",
                    "estimate": neyman.estimate,
                    "se": float(draws.std(ddof=1)),
                    "ci_low": bca[0],
                    "ci_high": bca[1],
                },
            ]
            # A relative change is undefined when the comparison mean is zero, in the data or in any
            # resample; it is then reported as NaN rather than computed on a subset of resamples.
            control_mean, control_draws = float(y_c[:, k].mean()), boot[c_arm][:, k]
            if control_mean > 0 and (control_draws > 0).all():
                relative = neyman.estimate / control_mean
                low, high = percentile_interval(draws / control_draws, ci_level)
            else:
                relative = low = high = np.nan
            lift_rows.append(
                {
                    **common,
                    "control_mean": control_mean,
                    "relative_lift": relative,
                    "ci_low": low,
                    "ci_high": high,
                    "interval": "bootstrap percentile",
                }
            )
            if outcome in plan["outcomes"]["binary"]:
                binary_rows.append({**common, **binary_contrast(y_t[:, k], y_c[:, k], ci_level)})

        diagnostics = assignment_diagnostics(
            covariates.loc[pair],
            treated_in_pair,
            plan["diagnostics"]["classifier_folds"],
            stream_int_seed(seed, plan["randomness"]["classifier_stream"]),
        )
        diagnostic_rows.append({"contrast": contrast["name"], **diagnostics})

    effects = pd.DataFrame(effect_rows)
    effects["p_holm"] = np.nan
    for outcome in outcomes:
        mask = (
            (effects["estimator"] == "difference_in_means")
            & (effects["role"] == "co-primary")
            & (effects["outcome"] == outcome)
        )
        effects.loc[mask, "p_holm"] = holm(effects.loc[mask, "p_value"].tolist())

    return ExperimentResults(
        effects=effects,
        binary=pd.DataFrame(binary_rows),
        relative_lift=pd.DataFrame(lift_rows),
        bootstrap_draws=pd.DataFrame(draw_columns),
        diagnostics=pd.DataFrame(diagnostic_rows),
        plan=plan,
    )
