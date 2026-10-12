"""Report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from causeway import schema  # noqa: E402
from causeway.features import COVARIATE_LABELS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ARM_COLORS = {"Mens E-Mail": "#2a78d6", "Womens E-Mail": "#eb6834", "No E-Mail": "#1baf7a"}
CONTRAST_COLORS = {
    "mens_vs_control": ARM_COLORS["Mens E-Mail"],
    "womens_vs_control": ARM_COLORS["Womens E-Mail"],
    "mens_vs_womens": INK_SECONDARY,
}
CONTRAST_LABELS = {
    "mens_vs_control": "Mens vs No E-Mail",
    "womens_vs_control": "Womens vs No E-Mail",
    "mens_vs_womens": "Mens vs Womens (secondary)",
}
ESTIMATOR_LABELS = {
    "difference_in_means": "Difference in means",
    "lin_adjusted": "Lin regression adjustment",
    "bootstrap_bca": "Bootstrap BCa",
    "bootstrap_percentile": "Bootstrap percentile",
}
EFFECT_AXIS = {
    "spend": "Effect on spend ($ per customer)",
    "visit": "Effect on visit rate (percentage points)",
    "conversion": "Effect on conversion rate (percentage points)",
}
OUTCOME_AXIS = {
    "spend": ("Mean two-week spend ($)", 1.0, "${:.2f}"),
    "visit": ("Visit rate (percentage points)", 100.0, "{:.1f}"),
    "conversion": ("Conversion rate (percentage points)", 100.0, "{:.2f}"),
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "legend.frameon": False,
            "text.parse_math": False,  # dollar signs are currency, not math
        }
    )


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def _title(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.0, ha="left", fontweight="bold", fontsize=10)


def _footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7.5, color=INK_MUTED, wrap=True)


def outcome_levels(frame: pd.DataFrame, path: Path, ci_level: float = 0.95) -> Path:
    """How do mean spend, visit rate and conversion rate differ across the randomized arms?"""
    apply_style()
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, outcome in zip(axes, ["spend", "visit", "conversion"], strict=True):
        label, scale, fmt = OUTCOME_AXIS[outcome]
        for i, arm in enumerate(schema.ARMS):
            y = frame.loc[frame["segment"] == arm, outcome].astype(float)
            mean, se = y.mean() * scale, y.std(ddof=1) / np.sqrt(len(y)) * scale
            ax.errorbar(mean, i, xerr=z * se, fmt="o", color=ARM_COLORS[arm], ms=7, lw=2, capsize=0)
            ax.annotate(
                fmt.format(mean),
                (mean + z * se, i),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK_SECONDARY,
            )
        ax.set_yticks(range(len(schema.ARMS)), schema.ARMS if outcome == "spend" else [""] * 3)
        ax.set_ylim(-0.6, len(schema.ARMS) - 0.4)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(label if outcome == "spend" else label.replace(" (percentage points)", " (%)"))
        ax.margins(x=0.25)
    _title(fig, "Outcome levels by randomized arm (mean with 95% CI)")
    counts = frame["segment"].astype(str).value_counts()
    _footnote(
        fig,
        "Two-week window after the campaign. "
        + ", ".join(f"n = {counts[arm]:,} ({arm})" for arm in schema.ARMS)
        + ". Intervals: normal approximation.",
    )
    fig.tight_layout()
    return save(fig, path)


def spend_among_converters(frame: pd.DataFrame, path: Path, maximum: float) -> Path:
    """What does spend look like among purchasers, and is there structure at the $499 maximum?"""
    apply_style()
    converters = frame[frame["conversion"] == 1]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
    for arm in schema.ARMS:
        values = np.sort(converters.loc[converters["segment"] == arm, "spend"].to_numpy())
        ecdf = np.arange(1, len(values) + 1) / len(values)
        left.step(values, ecdf, where="post", color=ARM_COLORS[arm], lw=2, label=f"{arm} (n = {len(values)})")
    left.axvline(maximum, color=INK_MUTED, lw=1, ls="--")
    left.annotate(
        f"observed maximum ${maximum:.0f}",
        (maximum, 0.55),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color=INK_SECONDARY,
    )
    left.set_xlabel("Two-week spend among purchasers ($)")
    left.set_ylabel("Cumulative share of purchasers")
    left.set_title("Spend distribution among purchasers, by arm")
    left.legend(loc="lower right", fontsize=7.5)

    edges = np.arange(0, maximum + 20, 20.0)
    counts, _ = np.histogram(converters.loc[converters["spend"] < maximum, "spend"], bins=edges)
    at_max = int((converters["spend"] == maximum).sum())
    right.bar(
        edges[:-1],
        counts,
        width=18,
        align="edge",
        color=AXIS,
        edgecolor=SURFACE,
        label=f"below ${maximum:.0f}",
    )
    right.bar(maximum, at_max, width=6, color=INK, label=f"exactly ${maximum:.0f} (n = {at_max})")
    right.set_xlabel("Two-week spend among purchasers ($, $20 bins)")
    right.set_ylabel("Purchasers")
    right.set_title("Upper tail: purchasers per spend band")
    right.legend(loc="upper right", fontsize=7.5)
    right.grid(axis="x", visible=False)
    _footnote(
        fig,
        "Descriptive only: conditioning on purchase conditions on an outcome of assignment, so these "
        "distributions are not causal comparisons. Whether $499 is a true maximum or a data-generation "
        "constraint is unknown.",
    )
    fig.tight_layout()
    return save(fig, path)


def love_plot(balance: pd.DataFrame, path: Path, threshold: float = 0.1, title: str | None = None) -> Path:
    """Are pre-treatment covariates balanced across arms? (standardized mean differences)"""
    apply_style()
    covariates = list(dict.fromkeys(balance["covariate"]))
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(covariates) + 1.2))
    ax.axvspan(-threshold, threshold, color=GRID, alpha=0.5, lw=0)
    ax.axvline(0, color=AXIS, lw=1)
    offsets = {name: off for name, off in zip(balance["comparison"].unique(), (-0.12, 0.12), strict=False)}
    for comparison, group in balance.groupby("comparison", sort=False):
        y = [covariates.index(c) + offsets[comparison] for c in group["covariate"]]
        ax.scatter(
            group["smd"],
            y,
            s=36,
            color=group["color"].iloc[0],
            edgecolor=SURFACE,
            linewidth=1,
            label=comparison,
            zorder=3,
        )
    ax.set_yticks(range(len(covariates)), [COVARIATE_LABELS.get(c, c) for c in covariates])
    ax.invert_yaxis()
    limit = max(threshold * 1.5, float(balance["smd"].abs().max()) * 1.2)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Standardized mean difference (treated arm minus No E-Mail)")
    ax.grid(axis="y", visible=False)
    _title(fig, title or "Covariate balance under randomization")
    ax.legend(loc="lower right", fontsize=7.5)
    _footnote(fig, f"Shaded band: |SMD| < {threshold}, the conventional reference for adequate balance.")
    fig.tight_layout()
    return save(fig, path)


def effect_forest(effects: pd.DataFrame, path: Path, estimators: list[str]) -> Path:
    """What is the ITT effect of assignment, and do the pre-specified estimators agree?"""
    apply_style()
    outcomes = ["spend", "visit", "conversion"]
    contrasts = list(dict.fromkeys(effects["contrast"]))
    rows = [(c, e) for c in contrasts for e in estimators]
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.3 * len(rows) + 1.6), sharey=True)
    for ax, outcome in zip(axes, outcomes, strict=True):
        scale = OUTCOME_AXIS[outcome][1]
        ax.axvline(0, color=AXIS, lw=1)
        for i, (contrast, estimator) in enumerate(rows):
            row = effects[
                (effects["contrast"] == contrast)
                & (effects["outcome"] == outcome)
                & (effects["estimator"] == estimator)
            ].iloc[0]
            color = CONTRAST_COLORS[contrast]
            ax.plot(
                [row["ci_low"] * scale, row["ci_high"] * scale],
                [i, i],
                color=color,
                lw=2,
                solid_capstyle="round",
            )
            ax.plot(row["estimate"] * scale, i, "o", color=color, ms=6, mec=SURFACE, mew=1)
        ax.set_xlabel("$ per customer" if outcome == "spend" else "Percentage points")
        ax.grid(axis="y", visible=False)
        ax.set_title(outcome.capitalize() + (" (primary)" if outcome == "spend" else " (secondary)"))
    axes[0].set_yticks(
        range(len(rows)), [f"{CONTRAST_LABELS[c]} · {ESTIMATOR_LABELS[e]}" for c, e in rows], fontsize=7.5
    )
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    for boundary in range(len(estimators), len(rows), len(estimators)):
        for ax in axes:
            ax.axhline(boundary - 0.5, color=GRID, lw=0.8)
    _title(fig, "Intention-to-treat effect of assignment to an e-mail campaign (point estimate, 95% CI)")
    _footnote(
        fig,
        "Estimand: effect of assignment, not of receiving or opening an e-mail. Co-primary: Mens and "
        "Womens vs No E-Mail (Holm-adjusted); Mens vs Womens is secondary.",
    )
    fig.tight_layout()
    return save(fig, path)


def bootstrap_distributions(
    draws: pd.DataFrame, effects: pd.DataFrame, path: Path, outcome: str = "spend"
) -> Path:
    """Is the normal approximation adequate for the skewed spend outcome?"""
    apply_style()
    contrasts = [c for c in CONTRAST_LABELS if f"{c}__{outcome}" in draws]
    fig, axes = plt.subplots(1, len(contrasts), figsize=(3.6 * len(contrasts), 2.9), sharey=True)
    for ax, contrast in zip(np.atleast_1d(axes), contrasts, strict=True):
        values = draws[f"{contrast}__{outcome}"].to_numpy()
        sel = effects[(effects["contrast"] == contrast) & (effects["outcome"] == outcome)].set_index(
            "estimator"
        )
        color = CONTRAST_COLORS[contrast]
        ax.hist(values, bins=60, density=True, color=color, alpha=0.35, edgecolor=SURFACE, linewidth=0.3)
        grid = np.linspace(values.min(), values.max(), 300)
        neyman = sel.loc["difference_in_means"]
        ax.plot(
            grid,
            stats.norm.pdf(grid, neyman["estimate"], neyman["se"]),
            color=INK,
            lw=1.5,
            label="Normal approximation (Neyman SE)",
        )
        for name, style in (("difference_in_means", "-"), ("bootstrap_bca", ":")):
            for bound in ("ci_low", "ci_high"):
                ax.axvline(sel.loc[name, bound], color=INK_SECONDARY, lw=1, ls=style)
        ax.set_title(CONTRAST_LABELS[contrast], fontsize=9)
        ax.set_xlabel("Bootstrap difference in mean spend ($)")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Density")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=7)
    _title(fig, "Bootstrap distributions of the spend effect (5,000 resamples within arms)")
    _footnote(fig, "Solid vertical lines: 95% normal-theory interval. Dotted: 95% BCa bootstrap interval.")
    fig.tight_layout()
    return save(fig, path)


SCENARIO_STYLE = {
    "randomized": {"label": "Randomized pair (no selection)", "color": INK_MUTED, "filled": False},
    "A": {"label": "Scenario A: moderate selection", "color": INK, "filled": True},
    "B": {"label": "Scenario B: strong selection", "color": "#4a3aa7", "filled": True},
}
ORACLE_COLOR = "#4a3aa7"
STATUS_CRITICAL = "#d03b3b"  # reserved status colour; always explained in the figure text


def selection_love_plot(balance: pd.DataFrame, path: Path, threshold: float = 0.1) -> Path:
    """How much covariate imbalance does the constructed selection create? (fixed display draw)"""
    apply_style()
    comparisons = list(dict.fromkeys(balance["comparison"]))
    covariates = list(dict.fromkeys(balance["covariate"]))
    fig, axes = plt.subplots(1, len(comparisons), figsize=(10, 0.32 * len(covariates) + 1.6), sharey=True)
    offsets = {"randomized": -0.2, "A": 0.0, "B": 0.2}
    for ax, comparison in zip(np.atleast_1d(axes), comparisons, strict=True):
        ax.axvspan(-threshold, threshold, color=GRID, alpha=0.5, lw=0)
        ax.axvline(0, color=AXIS, lw=1)
        sub = balance[balance["comparison"] == comparison]
        for state, style in SCENARIO_STYLE.items():
            rows = sub[sub["state"] == state]
            y = [covariates.index(c) + offsets[state] for c in rows["covariate"]]
            ax.scatter(
                rows["smd"],
                y,
                s=34,
                label=style["label"],
                zorder=3,
                linewidth=1.2,
                facecolor=style["color"] if style["filled"] else SURFACE,
                edgecolor=style["color"],
            )
        ax.set_title(CONTRAST_LABELS[comparison], fontsize=9)
        ax.set_xlabel("Standardized mean difference (treated minus control)")
        ax.grid(axis="y", visible=False)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(covariates)), [COVARIATE_LABELS.get(c, c) for c in covariates])
    first.invert_yaxis()
    handles, labels = first.get_legend_handles_labels()
    _title(fig, "Covariate imbalance created by the constructed selection (display draw 0)")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.95), ncol=3, fontsize=7.5)
    _footnote(
        fig,
        f"Shaded band: |SMD| < {threshold}. Selection uses past-year spend, recency, multichannel and "
        "newbie only; the other covariates become imbalanced through their correlation with these.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, path)


def design_propensity_overlap(panels: list[dict], path: Path) -> Path:
    """Do treated and control customers overlap in the true design propensity e*(x)?"""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True)
    bins = np.linspace(0, 1, 41)
    for ax, panel in zip(axes.ravel(), panels, strict=True):
        treated_arm = "Mens E-Mail" if panel["comparison"].startswith("mens") else "Womens E-Mail"
        for label, values, color in (
            (f"{treated_arm} (retained)", panel["e_treated"], ARM_COLORS[treated_arm]),
            ("No E-Mail (retained)", panel["e_control"], ARM_COLORS["No E-Mail"]),
        ):
            ax.hist(values, bins=bins, histtype="step", lw=2, color=color, label=label)
        ax.set_title(f"{CONTRAST_LABELS[panel['comparison']]} · Scenario {panel['scenario']}", fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("True design propensity e*(x)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Retained customers")
    _title(fig, "Overlap of the design propensity in the retained sample (display draw 0)")
    _footnote(
        fig,
        "e*(x) is the propensity fixed by the benchmark design, known only because the selection was "
        "constructed. Estimated propensities are in the propensity report.",
    )
    fig.tight_layout()
    return save(fig, path)


def naive_vs_oracle(summary_draws: pd.DataFrame, path: Path) -> Path:
    """How biased is the naive comparison under the constructed selection, and does the oracle remove it?"""
    apply_style()
    keys = summary_draws[["comparison", "scenario", "outcome"]].drop_duplicates().itertuples(index=False)
    rows = list(keys)
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(rows) + 1.4))
    ax.axvline(0, color=AXIS, lw=1)
    for i, (comparison, scenario, outcome) in enumerate(rows):
        sub = summary_draws[
            (summary_draws["comparison"] == comparison)
            & (summary_draws["scenario"] == scenario)
            & (summary_draws["outcome"] == outcome)
        ]
        for estimator, color, offset in (
            ("naive_difference", INK_SECONDARY, -0.15),
            ("oracle_ipw", ORACLE_COLOR, 0.15),
        ):
            rel = sub.loc[sub["estimator"] == estimator, "relative_error"] * 100
            low, mid, high = np.percentile(rel, [2.5, 50, 97.5])
            ax.plot([low, high], [i + offset] * 2, color=color, lw=2, solid_capstyle="round")
            ax.plot(
                mid,
                i + offset,
                "o",
                color=color,
                ms=5,
                mec=SURFACE,
                mew=1,
                label={"naive_difference": "Naive difference", "oracle_ipw": "Oracle IPW (true e*)"}[
                    estimator
                ]
                if i == 0
                else None,
            )
    ax.set_yticks(
        range(len(rows)),
        [f"{CONTRAST_LABELS[c].split(' (')[0]} · {s} · {o}" for c, s, o in rows],
        fontsize=7.5,
    )
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("Error relative to the experimental ATE benchmark (% of benchmark)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7.5)
    _title(fig, "Naive and oracle estimates across 200 constructed samples (median, 2.5-97.5 percentiles)")
    _footnote(
        fig,
        "Benchmark: difference in means on the full randomized pair. The oracle uses the true design "
        "propensity and checks the construction; it is not available to an analyst.",
    )
    fig.tight_layout()
    return save(fig, path)


ESTIMATOR_DISPLAY = {
    "naive_difference": "Naive difference",
    "oracle_ipw": "Oracle IPW (true e*)",
    "ipw_ate": "IPW (ATE)",
    "ipw_att": "IPW (ATT)",
    "matching_nn": "1:1 NN matching (ATT)",
    "outcome_regression": "Outcome regression (OLS)",
    "aipw": "AIPW (linear, cross-fitted)",
    "dml_irm": "DML (gradient boosting)",
}
ADJUSTMENT_STYLE = {
    "unadjusted": {"label": "Unadjusted", "color": INK_MUTED, "filled": False},
    "ipw_ate": {"label": "IPW (ATE weights)", "color": INK, "filled": True},
    "matched": {"label": "Matched sample", "color": "#4a3aa7", "filled": True},
}


def estimated_propensity_overlap(panels: list[dict], path: Path) -> Path:
    """Where do estimated propensities of treated and control customers overlap, by scenario?"""
    apply_style()
    scenarios = list(dict.fromkeys(p["scenario"] for p in panels))
    comparisons = list(dict.fromkeys(p["comparison"] for p in panels))
    fig, axes = plt.subplots(len(comparisons), len(scenarios), figsize=(11, 5.2), sharex=True)
    bins = np.linspace(0, 1, 41)
    for panel in panels:
        ax = axes[comparisons.index(panel["comparison"]), scenarios.index(panel["scenario"])]
        treated_arm = "Mens E-Mail" if panel["comparison"].startswith("mens") else "Womens E-Mail"
        ax.hist(
            panel["score_treated"],
            bins=bins,
            histtype="step",
            lw=2,
            color=ARM_COLORS[treated_arm],
            label=treated_arm,
        )
        ax.hist(
            panel["score_control"],
            bins=bins,
            histtype="step",
            lw=2,
            color=ARM_COLORS["No E-Mail"],
            label="No E-Mail",
        )
        ax.set_title(f"{CONTRAST_LABELS[panel['comparison']]} · Scenario {panel['scenario']}", fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", fontsize=6.5)
    for ax in axes[-1]:
        ax.set_xlabel("Estimated propensity score (logistic model)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Retained customers")
    _title(fig, "Estimated propensity overlap in the retained sample (display draw 0)")
    _footnote(
        fig, "Scenario C's model omits recency, so its scores understate how separated the arms really are."
    )
    fig.tight_layout()
    return save(fig, path)


def adjustment_love_plot(balance: pd.DataFrame, path: Path, threshold: float = 0.1) -> Path:
    """Does weighting or matching remove the imbalance, including on a hidden covariate?"""
    apply_style()
    panels = balance[["comparison", "scenario"]].drop_duplicates().itertuples(index=False)
    panels = list(panels)
    covariates = list(dict.fromkeys(balance["covariate"]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 0.55 * len(covariates) + 2.4), sharey=True)
    offsets = {"unadjusted": -0.2, "ipw_ate": 0.0, "matched": 0.2}
    for ax, (comparison, scenario) in zip(axes.ravel(), panels, strict=True):
        ax.axvspan(-threshold, threshold, color=GRID, alpha=0.5, lw=0)
        ax.axvline(0, color=AXIS, lw=1)
        sub = balance[(balance["comparison"] == comparison) & (balance["scenario"] == scenario)]
        for state, style in ADJUSTMENT_STYLE.items():
            rows = sub[sub["state"] == state]
            y = [covariates.index(c) + offsets[state] for c in rows["covariate"]]
            ax.scatter(
                rows["smd"],
                y,
                s=26,
                label=style["label"],
                zorder=3,
                linewidth=1.1,
                facecolor=style["color"] if style["filled"] else SURFACE,
                edgecolor=style["color"],
            )
        title = f"{CONTRAST_LABELS[comparison]} · Scenario {scenario}"
        ax.set_title(title + (" (recency hidden)" if scenario == "C" else ""), fontsize=8.5)
        ax.grid(axis="y", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("Standardized mean difference (treated minus control)")
    axes[0, 0].set_yticks(range(len(covariates)), [COVARIATE_LABELS.get(c, c) for c in covariates])
    axes[0, 0].invert_yaxis()
    for ax in axes[:, 0]:
        ax.tick_params(axis="y", labelsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _title(fig, "Covariate balance before and after propensity adjustment (display draw 0)")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.955), ncol=3, fontsize=7.5)
    _footnote(
        fig,
        f"Shaded band: |SMD| < {threshold}. SMDs use the unadjusted pooled SD as the scale. Balance on "
        "a covariate the model cannot see is not improved by adjusting for the others.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, path)


def weight_distributions(panels: list[dict], path: Path) -> Path:
    """How extreme are the ATE weights, and how much effective sample size do they leave?"""
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 2.9), sharey=True)
    bins = np.logspace(-2, 3, 50)
    for ax, panel in zip(np.atleast_1d(axes), panels, strict=True):
        treated_arm = "Mens E-Mail" if panel["comparison"].startswith("mens") else "Womens E-Mail"
        for label, values, color, ess in (
            (treated_arm, panel["w_treated"], ARM_COLORS[treated_arm], panel["ess_treated"]),
            ("No E-Mail", panel["w_control"], ARM_COLORS["No E-Mail"], panel["ess_control"]),
        ):
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                lw=2,
                color=color,
                label=f"{label}: ESS {ess:.0%} of n, max {values.max():.0f}x mean",
            )
        ax.set_xscale("log")
        # Plain tick labels: mathtext is disabled globally so that "$" renders as currency.
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        upper = max(panel["w_treated"].max(), panel["w_control"].max())
        ax.set_xlim(0.1, upper * 1.5)
        ax.set_title(f"{CONTRAST_LABELS[panel['comparison']]} · Scenario {panel['scenario']}", fontsize=8.5)
        ax.set_xlabel("ATE weight / mean weight in arm (log scale)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", fontsize=6.5)
    first = np.atleast_1d(axes)[0]
    first.set_ylim(0, first.get_ylim()[1] * 1.35)
    first.set_ylabel("Retained customers")
    _title(fig, "Inverse probability weights (display draw 0)")
    _footnote(
        fig,
        "ESS: Kish effective sample size as a share of the arm's retained customers. No weights were "
        "trimmed or clipped in the primary analysis.",
    )
    fig.tight_layout()
    return save(fig, path)


def estimator_error_grid(relative: pd.DataFrame, outcome: str, path: Path, estimators: list[str]) -> Path:
    """How close does each estimator get to its estimand-matched experimental benchmark?"""
    apply_style()
    scenarios = ["A", "B", "C"]
    comparisons = list(dict.fromkeys(relative["comparison"]))
    rows = [(c, e) for c in comparisons for e in estimators]
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.3 * len(rows) + 1.8), sharey=True)
    for ax, scenario in zip(axes, scenarios, strict=True):
        ax.axvline(0, color=AXIS, lw=1)
        for i, (comparison, estimator) in enumerate(rows):
            values = relative[
                (relative["comparison"] == comparison)
                & (relative["scenario"] == scenario)
                & (relative["outcome"] == outcome)
                & (relative["estimator"] == estimator)
            ]
            errors = values["relative_error"].dropna().to_numpy() * 100
            if not len(errors):
                continue
            low, mid, high = np.percentile(errors, [2.5, 50, 97.5])
            color = (
                ORACLE_COLOR
                if estimator == "oracle_ipw"
                else (INK_MUTED if estimator == "naive_difference" else CONTRAST_COLORS[comparison])
            )
            ax.plot([low, high], [i, i], color=color, lw=2, solid_capstyle="round")
            ax.plot(mid, i, "o", color=color, ms=5, mec=SURFACE, mew=1)
        ax.set_title(
            {"A": "Scenario A: moderate", "B": "Scenario B: strong", "C": "Scenario C: recency hidden"}[
                scenario
            ],
            fontsize=9,
        )
        ax.set_xlabel("Error (% of benchmark)")
        ax.grid(axis="y", visible=False)
        for boundary in range(len(estimators), len(rows), len(estimators)):
            ax.axhline(boundary - 0.5, color=GRID, lw=0.8)
    axes[0].set_yticks(
        range(len(rows)),
        [f"{CONTRAST_LABELS[c].split(' (')[0]} · {ESTIMATOR_DISPLAY[e]}" for c, e in rows],
        fontsize=7.5,
    )
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    _title(
        fig,
        f"{outcome.capitalize()}: estimator error across 200 constructed samples (median, 2.5-97.5 "
        "percentiles)",
    )
    by_benchmark: dict[str, list[str]] = {}
    for estimator in estimators:
        benchmark = relative.loc[relative["estimator"] == estimator, "benchmark"].iloc[0]
        by_benchmark.setdefault(benchmark, []).append(ESTIMATOR_DISPLAY[estimator])
    _footnote(
        fig,
        "Each estimator is compared with the experimental benchmark for its own estimand. "
        + "; ".join(f"{benchmark} benchmark: {', '.join(names)}" for benchmark, names in by_benchmark.items())
        + ".",
    )
    fig.tight_layout()
    return save(fig, path)


COMPARISON_GROUPS = {
    "ATE": [
        "experiment",
        "naive_difference",
        "oracle_ipw",
        "outcome_regression",
        "ipw_ate",
        "aipw",
        "dml_irm",
    ],
    "ATT": ["experiment_att", "ipw_att", "matching_nn"],
}
EXPERIMENT_LABELS = {
    "experiment": "Randomized experiment (ATE)",
    "experiment_att": "Experimental ATT benchmark",
}


def comparison_forest(table: pd.DataFrame, outcome: str, path: Path) -> Path:
    """On one realized observational dataset, which estimates land near their own experimental benchmark?"""
    apply_style()
    comparisons = list(dict.fromkeys(table["comparison"]))
    scenarios = ["A", "B", "C"]
    rows = [(group, name) for group, names in COMPARISON_GROUPS.items() for name in names]
    scale = OUTCOME_AXIS[outcome][1]
    fig, axes = plt.subplots(
        len(comparisons), 3, figsize=(11.5, 0.27 * len(rows) * len(comparisons) + 2.2), sharey=True
    )
    for r, comparison in enumerate(comparisons):
        for c, scenario in enumerate(scenarios):
            ax = axes[r, c]
            cell = table[
                (table["comparison"] == comparison)
                & (table["scenario"] == scenario)
                & (table["outcome"] == outcome)
            ].set_index("estimator")
            for group, style in (("ATE", "-"), ("ATT", "--")):
                members = [i for i, (g, _) in enumerate(rows) if g == group]
                value = (
                    cell.loc["experiment" if group == "ATE" else "experiment_att", "benchmark_value"] * scale
                )
                ax.plot(
                    [value, value],
                    [members[0] - 0.45, members[-1] + 0.45],
                    color=INK_SECONDARY,
                    lw=1,
                    ls=style,
                )
            ax.axvline(0, color=AXIS, lw=0.8)
            for i, (_, name) in enumerate(rows):
                row = cell.loc[name]
                if name.startswith("experiment"):
                    color = INK
                elif name == "naive_difference":
                    color = INK_MUTED
                elif name == "oracle_ipw":
                    color = ORACLE_COLOR
                else:
                    color = CONTRAST_COLORS[comparison]
                ax.plot(
                    [row["ci_low"] * scale, row["ci_high"] * scale],
                    [i, i],
                    color=color,
                    lw=2,
                    solid_capstyle="round",
                )
                ax.plot(
                    row["estimate"] * scale,
                    i,
                    "D" if name.startswith("experiment") else "o",
                    color=color,
                    ms=5,
                    mec=SURFACE,
                    mew=1,
                )
            ax.axhline(len(COMPARISON_GROUPS["ATE"]) - 0.5, color=GRID, lw=0.8)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{CONTRAST_LABELS[comparison]} · Scenario {scenario}", fontsize=8.5)
            if r == len(comparisons) - 1:
                ax.set_xlabel(EFFECT_AXIS[outcome])
    labels = [EXPERIMENT_LABELS.get(name, ESTIMATOR_DISPLAY.get(name, name)) for _, name in rows]
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(rows)), labels, fontsize=7.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
    _title(
        fig, f"{outcome.capitalize()}: estimates from one constructed observational dataset (display draw 0)"
    )
    _footnote(
        fig,
        "Upper block: estimators of the ATE, solid line = experimental ATE. Lower block: estimators of the "
        "ATT, dashed line = experimental ATT (randomized pair weighted by e*). Intervals: 95%. The "
        "display draw is fixed by the frozen specification, not chosen for appearance.",
    )
    fig.tight_layout()
    return save(fig, path)


def bias_rmse_summary(performance: pd.DataFrame, outcome: str, path: Path, estimators: list[str]) -> Path:
    """Across 200 draws, how large is each estimator's bias relative to its total error (RMSE)?"""
    apply_style()
    comparisons = list(dict.fromkeys(performance["comparison"]))
    rows = [(c, e) for c in comparisons for e in estimators]
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.3 * len(rows) + 1.9), sharey=True)
    for ax, scenario in zip(axes, ["A", "B", "C"], strict=True):
        ax.axvline(0, color=AXIS, lw=1)
        for i, (comparison, estimator) in enumerate(rows):
            row = performance[
                (performance["comparison"] == comparison)
                & (performance["scenario"] == scenario)
                & (performance["outcome"] == outcome)
                & (performance["estimator"] == estimator)
            ].iloc[0]
            benchmark = row["benchmark_value"]
            rmse = 100 * row["rmse"] / abs(benchmark)
            bias = 100 * row["bias"] / benchmark
            half = 100 * 2 * row["bias_mc_se"] / abs(benchmark)
            ax.barh(i, rmse, height=0.55, color=GRID, edgecolor=SURFACE)
            color = INK if row["unbiased_within_mc_error"] else STATUS_CRITICAL
            ax.plot([bias - half, bias + half], [i, i], color=color, lw=2, solid_capstyle="round")
            ax.plot(bias, i, "o", color=color, ms=5, mec=SURFACE, mew=1)
        ax.set_title(
            {"A": "Scenario A: moderate", "B": "Scenario B: strong", "C": "Scenario C: recency hidden"}[
                scenario
            ],
            fontsize=9,
        )
        ax.set_xlabel("% of matched benchmark")
        ax.grid(axis="y", visible=False)
        for boundary in range(len(estimators), len(rows), len(estimators)):
            ax.axhline(boundary - 0.5, color=GRID, lw=0.8)
    axes[0].set_yticks(
        range(len(rows)),
        [f"{CONTRAST_LABELS[c].split(' (')[0]} · {ESTIMATOR_DISPLAY[e]}" for c, e in rows],
        fontsize=7.5,
    )
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    _title(fig, f"{outcome.capitalize()}: bias (dot, ±2 Monte Carlo SE) and RMSE (bar) across 200 draws")
    _footnote(
        fig,
        "Red: bias larger than twice its Monte Carlo standard error. Grey bar: RMSE, which adds the "
        "estimator's sampling variability to its bias. Each estimator against its own estimand's benchmark.",
    )
    fig.tight_layout()
    return save(fig, path)


CLASSIFICATION_MARKER = {"benefit": "o", "harm": "s", "inconclusive": "o"}


def gates_plot(gates: pd.DataFrame, verdicts: pd.DataFrame, path: Path) -> Path:
    """Do customers with larger predicted effects actually show larger effects on held-out data?"""
    apply_style()
    outcomes = list(dict.fromkeys(gates["outcome"]))
    contrasts = list(dict.fromkeys(gates["contrast"]))
    fig, axes = plt.subplots(
        len(outcomes), len(contrasts), figsize=(10, 2.5 * len(outcomes) + 1), squeeze=False
    )
    for r, outcome in enumerate(outcomes):
        scale = OUTCOME_AXIS[outcome][1]
        for c, contrast in enumerate(contrasts):
            ax = axes[r, c]
            sub = gates[(gates["outcome"] == outcome) & (gates["contrast"] == contrast)].set_index(
                "parameter"
            )
            ate = sub.loc["b1"]
            ax.axhspan(ate["ci_low"] * scale, ate["ci_high"] * scale, color=GRID, alpha=0.6, lw=0)
            ax.axhline(ate["estimate"] * scale, color=INK_SECONDARY, lw=1)
            ax.axhline(0, color=AXIS, lw=0.8)
            for g in range(1, 6):
                row = sub.loc[f"gate_{g}"]
                ax.plot(
                    [g, g],
                    [row["ci_low"] * scale, row["ci_high"] * scale],
                    color=CONTRAST_COLORS[contrast],
                    lw=2,
                    solid_capstyle="round",
                )
                ax.plot(g, row["estimate"] * scale, "o", color=CONTRAST_COLORS[contrast], ms=6, mec=SURFACE)
            verdict = verdicts[(verdicts["outcome"] == outcome) & (verdicts["contrast"] == contrast)].iloc[0]
            ax.set_title(
                f"{CONTRAST_LABELS[contrast]} · {outcome}: BLP b2 = {verdict['b2']:.2f} "
                + ("(p < 0.001)" if verdict["b2_p_value"] < 0.001 else f"(p = {verdict['b2_p_value']:.3f})"),
                fontsize=8.5,
            )
            ax.set_xticks(range(1, 6), ["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
            ax.grid(axis="x", visible=False)
            if c == 0:
                ax.set_ylabel(EFFECT_AXIS[outcome])
    _title(fig, "Sorted group average effects (GATES) by quintile of predicted effect, held-out halves")
    _footnote(
        fig,
        "Medians over 100 random splits; intervals have at least 95% coverage. Grey band: the average "
        "effect (b1) with its interval. Flat GATES and b2 near 0 mean the predicted differences do not "
        "replicate on new data.",
    )
    fig.tight_layout()
    return save(fig, path)


SEGMENT_NAMES = {
    "history_segment": "Past-year spend",
    "recency_band": "Months since purchase",
    "channel": "Channel",
    "newbie": "New customer",
    "zip_code": "ZIP area",
    "mens": "Bought Mens merchandise",
    "womens": "Bought Womens merchandise",
}


def _segment_label(variable: str, level) -> str:
    level = str(level)
    if variable == "history_segment":
        level = level.split(") ", 1)[-1]  # "1) $0 - $100" -> "$0 - $100"
    elif level in ("0", "1"):
        level = "yes" if level == "1" else "no"
    return f"{SEGMENT_NAMES.get(variable, variable)}: {level}"


def _segment_rows(levels: pd.DataFrame) -> list[tuple]:
    """(variable, level) pairs in display order."""
    variables = list(dict.fromkeys(levels["variable"]))
    return sorted(
        levels[["variable", "level"]].drop_duplicates().itertuples(index=False),
        # recency bands are strings, so "10-12" would otherwise sort before "4-6"
        key=lambda r: (variables.index(r[0]), int(str(r[1]).split("-")[0]) if r[0] == "recency_band" else 0),
    )


def segment_forest(levels: pd.DataFrame, overall: dict[str, float], outcome: str, path: Path) -> Path:
    """Within each pre-specified segment, how large is the effect and how uncertain is it?"""
    apply_style()
    contrasts = list(dict.fromkeys(levels["contrast"]))
    rows = _segment_rows(levels)
    scale = OUTCOME_AXIS[outcome][1]
    fig, axes = plt.subplots(1, len(contrasts), figsize=(10, 0.24 * len(rows) + 1.8), sharey=True)
    for ax, contrast in zip(np.atleast_1d(axes), contrasts, strict=True):
        sub = levels[(levels["contrast"] == contrast) & (levels["outcome"] == outcome)]
        ax.axvline(0, color=AXIS, lw=1)
        ax.axvline(overall[contrast] * scale, color=INK_SECONDARY, lw=1, ls="--")
        for i, (variable, level) in enumerate(rows):
            row = sub[(sub["variable"] == variable) & (sub["level"] == level)].iloc[0]
            color = CONTRAST_COLORS[contrast] if row["classification"] != "harm" else STATUS_CRITICAL
            ax.plot(
                [row["ci_low"] * scale, row["ci_high"] * scale],
                [i, i],
                color=color,
                lw=1.8,
                solid_capstyle="round",
            )
            ax.plot(
                row["estimate"] * scale,
                i,
                CLASSIFICATION_MARKER[row["classification"]],
                ms=5,
                color=color if row["classification"] != "inconclusive" else SURFACE,
                markeredgecolor=color,
            )
        ax.set_title(CONTRAST_LABELS[contrast], fontsize=9)
        ax.set_xlabel(EFFECT_AXIS[outcome])
        ax.grid(axis="y", visible=False)
        for boundary in range(1, len(rows)):
            if rows[boundary][0] != rows[boundary - 1][0]:
                ax.axhline(boundary - 0.5, color=GRID, lw=0.8)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(rows)), [_segment_label(v, lv) for v, lv in rows], fontsize=7)
    first.set_ylim(len(rows) - 0.5, -0.5)
    _title(fig, f"{outcome.capitalize()}: effect within pre-specified segments (exploratory)")
    _footnote(
        fig,
        "Filled marker: BH q < 0.05 (benefit); hollow: inconclusive. Dashed line: overall effect. "
        "95% intervals, unadjusted for multiplicity; wide intervals are results, not noise to hide.",
    )
    fig.tight_layout()
    return save(fig, path)


def cate_distribution(panels: list[dict], path: Path) -> Path:
    """How widely do the forest's predicted individual effects spread, relative to their uncertainty?"""
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.9))
    for ax, panel in zip(np.atleast_1d(axes), panels, strict=True):
        _, scale, _ = OUTCOME_AXIS[panel["outcome"]]
        ax.hist(
            panel["cate"] * scale,
            bins=50,
            color=CONTRAST_COLORS[panel["contrast"]],
            alpha=0.5,
            edgecolor=SURFACE,
            linewidth=0.3,
        )
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_title(f"{CONTRAST_LABELS[panel['contrast']]} · {panel['outcome']}", fontsize=8.5)
        ax.set_xlabel("Predicted effect" + (" ($)" if panel["outcome"] == "spend" else " (pp)"))
        ax.grid(axis="x", visible=False)
        ax.annotate(
            f"pointwise 95% CI excludes 0:\n{panel['share_significant']:.1%} of customers",
            (0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7,
            color=INK_SECONDARY,
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    np.atleast_1d(axes)[0].set_ylabel("Customers")
    _title(fig, "Causal forest: predicted individual effects (descriptive)")
    _footnote(
        fig,
        "Spread in point predictions is not evidence of heterogeneity on its own; the formal test (GATES "
        "and BLP) decides. Pointwise intervals are not adjusted for testing every customer.",
    )
    fig.tight_layout()
    return save(fig, path)


def uplift_curves(curves: pd.DataFrame, path: Path, outcome: str) -> Path:
    """Does ranking customers by predicted effect beat random targeting of the same number of customers?"""
    apply_style()
    emails = list(dict.fromkeys(curves["email"]))
    _, scale, _ = OUTCOME_AXIS[outcome]
    fig, axes = plt.subplots(1, len(emails), figsize=(4.6 * len(emails), 3.1), sharey=True)
    for ax, email in zip(np.atleast_1d(axes), emails, strict=True):
        sub = curves[curves["email"] == email]
        arm = "Mens E-Mail" if email == "mens" else "Womens E-Mail"
        ax.plot(
            sub["fraction_targeted"] * 100,
            sub["model_incremental_value"] * scale,
            color=ARM_COLORS[arm],
            lw=2,
            label="Ranked by predicted effect",
        )
        ax.plot(
            sub["fraction_targeted"] * 100,
            sub["random_incremental_value"] * scale,
            color=INK_MUTED,
            lw=1.5,
            ls="--",
            label="Random selection",
        )
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(f"{arm} vs No E-Mail", fontsize=9)
        ax.set_xlabel("Customers e-mailed (%)")
        ax.legend(loc="upper left", fontsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel(
        "Incremental " + ("spend per customer ($)" if outcome == "spend" else "visits per customer (pp)")
    )
    _title(fig, f"Held-out uplift curves ({outcome})")
    _footnote(
        fig,
        "Estimated policy values from cross-fitted predictions and inverse probability weighting on the "
        "randomized data. These policies were never run. Point estimates; intervals are in the policy "
        "table.",
    )
    fig.tight_layout()
    return save(fig, path)


SPECIFICATION_LABELS = {
    "propensity=main": "main propensity",
    "propensity=alt_categorical": "categorical propensity",
    "propensity=alt_gbm": "boosted propensity",
    "outcome=alt_gbm": "boosted outcome model",
}


def experiment_robustness_plot(table: pd.DataFrame, path: Path) -> Path:
    """Does the spend effect depend on how the largest purchases or duplicate rows are handled?"""
    apply_style()
    dollars = table[~table["outcome"].str.startswith("log") & table["outcome"].str.startswith("spend")]
    contrasts = list(dict.fromkeys(dollars["contrast"]))
    labels = list(dict.fromkeys(zip(dollars["analysis"], dollars["outcome"], strict=True)))
    fig, axes = plt.subplots(1, len(contrasts), figsize=(11, 0.4 * len(labels) + 1.6), sharey=True)
    for ax, contrast in zip(axes, contrasts, strict=True):
        sub = dollars[dollars["contrast"] == contrast]
        primary = sub[(sub["analysis"] == "spend version") & (sub["outcome"] == "spend")].iloc[0]
        ax.axvline(0, color=AXIS, lw=1)
        ax.axvline(primary["estimate"], color=INK_SECONDARY, lw=1, ls="--")
        for i, (analysis, outcome) in enumerate(labels):
            row = sub[(sub["analysis"] == analysis) & (sub["outcome"] == outcome)].iloc[0]
            ax.plot(
                [row["ci_low"], row["ci_high"]],
                [i, i],
                color=CONTRAST_COLORS[contrast],
                lw=2,
                solid_capstyle="round",
            )
            ax.plot(row["estimate"], i, "o", color=CONTRAST_COLORS[contrast], ms=5, mec=SURFACE)
        ax.set_title(CONTRAST_LABELS[contrast], fontsize=9)
        ax.set_xlabel("Effect on spend ($ per customer)")
        ax.grid(axis="y", visible=False)
    names = [outcome if analysis == "spend version" else f"spend, {analysis}" for analysis, outcome in labels]
    axes[0].set_yticks(range(len(labels)), names, fontsize=7.5)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    _title(fig, "Robustness of the experimental spend effect")
    _footnote(
        fig,
        "Dashed line: primary estimate. No version drops customers based on what they spent. "
        "The log-scale version answers a different question and is in the table only.",
    )
    fig.tight_layout()
    return save(fig, path)


def specification_bias(summary: pd.DataFrame, outcome: str, path: Path) -> Path:
    """Do the Part II conclusions survive other propensity and outcome models?"""
    apply_style()
    rows = summary[(summary["outcome"] == outcome) & summary["specification"].isin(SPECIFICATION_LABELS)][
        ["comparison", "estimator", "specification"]
    ].drop_duplicates()
    order = {
        name: i for i, name in enumerate(["outcome_regression", "ipw_ate", "aipw", "ipw_att", "matching_nn"])
    }
    rows = rows.sort_values(
        ["comparison", "estimator", "specification"],
        key=lambda col: col.map(order) if col.name == "estimator" else col,
    )
    rows = list(rows.itertuples(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.26 * len(rows) + 1.9), sharey=True)
    for ax, scenario in zip(axes, ["A", "B", "C"], strict=True):
        ax.axvline(0, color=AXIS, lw=1)
        for i, (comparison, estimator, specification) in enumerate(rows):
            row = summary[
                (summary["comparison"] == comparison)
                & (summary["scenario"] == scenario)
                & (summary["outcome"] == outcome)
                & (summary["estimator"] == estimator)
                & (summary["specification"] == specification)
            ].iloc[0]
            scale = 100 / abs(row["benchmark_value"])
            ax.barh(i, row["rmse"] * scale, height=0.55, color=GRID, edgecolor=SURFACE)
            bias, half = row["bias"] * scale, 2 * row["bias_mc_se"] * scale
            color = INK if abs(row["bias"]) <= 2 * row["bias_mc_se"] else STATUS_CRITICAL
            ax.plot([bias - half, bias + half], [i, i], color=color, lw=2, solid_capstyle="round")
            ax.plot(bias, i, "o", color=color, ms=4.5, mec=SURFACE)
        ax.set_title(
            {"A": "Scenario A", "B": "Scenario B", "C": "Scenario C (recency hidden)"}[scenario], fontsize=9
        )
        ax.set_xlabel("% of matched benchmark")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(
        range(len(rows)),
        [
            f"{CONTRAST_LABELS[c].split(' vs')[0]} · {ESTIMATOR_DISPLAY[e]} · {SPECIFICATION_LABELS[s]}"
            for c, e, s in rows
        ],
        fontsize=6.8,
    )
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    _title(
        fig, f"{outcome.capitalize()}: bias (dot, ±2 Monte Carlo SE) and RMSE (bar) by model specification"
    )
    _footnote(fig, "Red: bias larger than twice its Monte Carlo SE. 200 draws per comparison and scenario.")
    fig.tight_layout()
    return save(fig, path)


def trimming_and_calipers(summary: pd.DataFrame, path: Path, outcome: str = "spend") -> Path:
    """What do trimming and tighter calipers buy, and what do they cost in discarded customers?"""
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.4))
    trims = summary[
        summary["specification"].str.startswith("trim")
        & (summary["outcome"] == outcome)
        & (summary["scenario"] == "B")
    ]
    bounds = sorted(trims["specification"].unique())
    for estimator, marker in (("ipw_ate", "o"), ("aipw", "s"), ("dml_irm", "D")):
        for comparison in dict.fromkeys(trims["comparison"]):
            sub = (
                trims[(trims["estimator"] == estimator) & (trims["comparison"] == comparison)]
                .set_index("specification")
                .loc[bounds]
            )
            left.plot(
                range(len(bounds)),
                100 * sub["rmse"] / sub["benchmark_value"].abs(),
                marker=marker,
                color=CONTRAST_COLORS[comparison],
                lw=1.5,
                label=f"{CONTRAST_LABELS[comparison].split(' vs')[0]} · {ESTIMATOR_DISPLAY[estimator]}",
            )
    left.set_xticks(range(len(bounds)), [b.removeprefix("trim=") for b in bounds])
    left.set_xlabel("Trimming bounds on the estimated propensity")
    left.set_ylabel("RMSE (% of trimmed benchmark)")
    left.set_title(f"Scenario B, {outcome}: error against the trimmed-population ATE", fontsize=9)
    left.legend(fontsize=6.5, loc="upper right")
    calipers = summary[summary["specification"].str.startswith("caliper") & (summary["outcome"] == outcome)]
    for comparison in dict.fromkeys(calipers["comparison"]):
        for scenario, style in (("A", "-"), ("B", "--")):
            sub = calipers[(calipers["comparison"] == comparison) & (calipers["scenario"] == scenario)]
            right.plot(
                100 * sub["mean_share_treated_discarded"],
                100 * sub["bias"] / sub["benchmark_value"],
                marker="o",
                ls=style,
                color=CONTRAST_COLORS[comparison],
                label=f"{CONTRAST_LABELS[comparison].split(' vs')[0]} · Scenario {scenario}",
            )
            for _, row in sub.iterrows():
                right.annotate(
                    row["specification"].removeprefix("caliper="),
                    (100 * row["mean_share_treated_discarded"], 100 * row["bias"] / row["benchmark_value"]),
                    xytext=(4, 2),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=INK_SECONDARY,
                )
    right.axhline(0, color=AXIS, lw=1)
    right.set_xlabel("Treated customers left unmatched (%)")
    right.set_ylabel("Bias (% of ATT benchmark)")
    right.set_title(f"1:1 matching, {outcome}: caliper (SD of logit) vs bias", fontsize=9)
    right.legend(fontsize=6.5, loc="best")
    _title(fig, "Trimming and caliper sensitivity")
    _footnote(
        fig,
        "Trimming changes the estimand to the trimmed population, so each trimmed estimate is compared "
        "with the experimental ATE over the same population (true e* within the bounds).",
    )
    fig.tight_layout()
    return save(fig, path)


def sensitivity_summary(sensitivity: pd.DataFrame, path: Path) -> Path:
    """Do standard sensitivity summaries look any different when a confounder really is hidden?"""
    apply_style()
    panels = [
        ("visit_rv_q1", "Robustness value, visit (partial R²)"),
        ("visit_e_value", "E-value, visit risk ratio"),
        ("visit_rosenbaum_gamma", "Rosenbaum Γ*, matched visit pairs"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    comparisons = list(dict.fromkeys(sensitivity["comparison"]))
    for ax, (column, title) in zip(axes, panels, strict=True):
        for c_index, comparison in enumerate(comparisons):
            for s_index, scenario in enumerate(["A", "C"]):
                values = sensitivity[
                    (sensitivity["comparison"] == comparison) & (sensitivity["scenario"] == scenario)
                ][column].dropna()
                if values.empty:  # e.g. Gamma* is undefined when no draw is significant at Gamma = 1
                    continue
                position = c_index * 2.5 + s_index
                low, mid, high = np.percentile(values, [2.5, 50, 97.5])
                ax.plot(
                    [position, position],
                    [low, high],
                    color=CONTRAST_COLORS[comparison],
                    lw=2,
                    solid_capstyle="round",
                )
                ax.plot(
                    position,
                    mid,
                    "o" if scenario == "A" else "s",
                    color=CONTRAST_COLORS[comparison],
                    ms=6,
                    mec=SURFACE,
                )
                if column == "visit_rv_q1" and scenario == "C":
                    hidden = sensitivity[
                        (sensitivity["comparison"] == comparison) & (sensitivity["scenario"] == "C")
                    ]["visit_hidden_recency_r2_yz"].median()
                    ax.plot(position, hidden, "x", color=INK, ms=7)
        ax.set_xticks([0, 1, 2.5, 3.5], ["Mens\nA", "Mens\nC", "Womens\nA", "Womens\nC"], fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    _title(fig, "Sensitivity summaries across 200 draws (median, 2.5–97.5 percentiles)")
    _footnote(
        fig,
        "Circles: Scenario A (no hidden confounder). Squares: Scenario C (recency hidden). Cross: the "
        "hidden recency's actual partial R² with visit.",
    )
    fig.tight_layout()
    return save(fig, path)


def recovery_heatmap(performance: pd.DataFrame, path: Path, estimators: list[str]) -> Path:
    """Which estimators recover their experimental benchmark, and under which kind of selection?"""
    apply_style()
    columns = [(c, s) for c in ("mens_vs_control", "womens_vs_control") for s in ("A", "B", "C")]
    outcomes = ["spend", "visit"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.36 * len(estimators) + 1.9), sharey=True)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("error", [SURFACE, "#b9b8af", "#55544f"])
    for ax, outcome in zip(axes, outcomes, strict=True):
        cells = performance[performance["outcome"] == outcome].set_index(
            ["estimator", "comparison", "scenario"]
        )[["bias_pct", "unbiased_within_mc_error"]]
        values = np.array([[cells.loc[(e, c, s), "bias_pct"] * 100 for c, s in columns] for e in estimators])
        ax.imshow(np.abs(values), cmap=ramp, vmin=0, vmax=60, aspect="auto")
        for i, estimator in enumerate(estimators):
            for j, (comparison, scenario) in enumerate(columns):
                clean = bool(cells.loc[(estimator, comparison, scenario), "unbiased_within_mc_error"])
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:+.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="normal" if clean else "bold",
                    color=INK if abs(values[i, j]) < 40 else SURFACE,
                )
        ax.set_xticks(
            range(len(columns)), [f"{c.split('_')[0].capitalize()}\n{s}" for c, s in columns], fontsize=7.5
        )
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(2.5, color=SURFACE, lw=3)
        ax.set_title(f"{outcome.capitalize()}: mean bias, % of matched benchmark", fontsize=9)
    axes[0].set_yticks(range(len(estimators)), [ESTIMATOR_DISPLAY[e] for e in estimators], fontsize=7.5)
    _title(fig, "Estimator recovery across 200 constructed samples per cell")
    _footnote(
        fig,
        "Columns: comparison and scenario (A moderate selection, B strong, C recency hidden). Bold: bias "
        "larger than twice its Monte Carlo SE. Shading darkens with |bias| and saturates at 60%. Each "
        "estimator is compared with the benchmark for its own estimand (ATE or ATT).",
    )
    fig.tight_layout()
    return save(fig, path)
