"""Static, publication-quality figures. Each function answers one stated analytical question.

Colour encodes identity only and follows the entity: each arm keeps one colour in every figure
(palette validated for colour-vision deficiency; values are always labelled as well).
"""

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
    fig.suptitle(
        "Outcome levels by randomized arm (mean with 95% CI)",
        x=0.0,
        ha="left",
        fontweight="bold",
        fontsize=10,
    )
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
    ax.set_title(title or "Covariate balance under randomization")
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
        label, scale, _ = OUTCOME_AXIS[outcome]
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
        ax.set_xlabel(
            label.replace("Mean two-week spend ($)", "Effect on two-week spend ($ per customer)").replace(
                " rate (percentage points)", " effect (percentage points)"
            )
        )
        ax.grid(axis="y", visible=False)
        ax.set_title(outcome.capitalize() + (" (primary)" if outcome == "spend" else " (secondary)"))
    axes[0].set_yticks(
        range(len(rows)), [f"{CONTRAST_LABELS[c]} · {ESTIMATOR_LABELS[e]}" for c, e in rows], fontsize=7.5
    )
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    for boundary in range(len(estimators), len(rows), len(estimators)):
        for ax in axes:
            ax.axhline(boundary - 0.5, color=GRID, lw=0.8)
    fig.suptitle(
        "Intention-to-treat effect of assignment to an e-mail campaign (point estimate, 95% CI)",
        x=0.0,
        ha="left",
        fontweight="bold",
        fontsize=10,
    )
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
    fig.suptitle(
        "Bootstrap distributions of the spend effect (5,000 resamples within arms)",
        x=0.0,
        ha="left",
        fontweight="bold",
        fontsize=10,
    )
    _footnote(fig, "Solid vertical lines: 95% normal-theory interval. Dotted: 95% BCa bootstrap interval.")
    fig.tight_layout()
    return save(fig, path)


SCENARIO_STYLE = {
    "randomized": {"label": "Randomized pair (no selection)", "color": INK_MUTED, "filled": False},
    "A": {"label": "Scenario A: moderate selection", "color": INK, "filled": True},
    "B": {"label": "Scenario B: strong selection", "color": "#4a3aa7", "filled": True},
}
ORACLE_COLOR = "#4a3aa7"


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
    fig.suptitle(
        "Covariate imbalance created by the constructed selection (display draw 0)",
        x=0.0,
        ha="left",
        fontweight="bold",
        fontsize=10,
    )
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
    fig.suptitle(
        "Overlap of the design propensity in the retained sample (display draw 0)",
        x=0.0,
        ha="left",
        fontweight="bold",
        fontsize=10,
    )
    _footnote(
        fig,
        "e*(x) is the propensity fixed by the benchmark design, known only because the selection was "
        "constructed. Estimated propensities are examined in Milestone 5.",
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
    ax.set_title("Naive and oracle estimates across 200 constructed samples (median, 2.5-97.5 percentiles)")
    _footnote(
        fig,
        "Benchmark: difference in means on the full randomized pair. The oracle uses the true design "
        "propensity and checks the construction; it is not available to an analyst.",
    )
    fig.tight_layout()
    return save(fig, path)
