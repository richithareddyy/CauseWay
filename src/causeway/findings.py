"""The final findings report: every milestone's saved results, read back and summarised in one place."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from causeway import visualization as viz
from causeway.comparison import check_provenance
from causeway.config import PROJECT_ROOT, DataPaths
from causeway.provenance import read_provenance
from causeway.reporting import ReportOutputs, _fmt_ci, _fmt_p, _save_tables

SOURCES = ["experiment", "benchmark", "propensity", "estimation", "comparison", "heterogeneity", "robustness"]
CONTRASTS = {"mens_vs_control": "Mens vs No E-Mail", "womens_vs_control": "Womens vs No E-Mail"}
RECOVERY_ESTIMATORS = [
    "naive_difference",
    "oracle_ipw",
    "outcome_regression",
    "ipw_ate",
    "aipw",
    "dml_irm",
    "ipw_att",
    "matching_nn",
]
TABLE_ESTIMATORS = ["experiment", "experiment_att", *RECOVERY_ESTIMATORS]

# (status, Part I, Part II, how it is examined)
ASSUMPTIONS = {
    "Randomized assignment": (
        "supported by design",
        "Holds: assignment was randomized by the data provider.",
        "Removed on purpose by the constructed selection.",
        "Covariate SMDs, a likelihood-ratio test and an out-of-fold classifier AUC for each arm.",
    ),
    "Unconfoundedness given the analyst's covariates": (
        "fundamentally untestable",
        "Not needed.",
        "True by construction in A and B, false in C (recency hidden).",
        "Cannot be checked from observed data; omitted-variable bounds, E-values and Rosenbaum bounds "
        "show how strong a hidden confounder would have to be.",
    ),
    "Positivity (overlap)": (
        "empirically diagnosable",
        "Holds: each arm received about a third of customers.",
        "Weakened in A, stressed in B.",
        "Propensity overlap plots, extreme-weight shares, effective sample size, trimming and calipers.",
    ),
    "Consistency (one version of each e-mail)": (
        "partially testable",
        "Assumed: each arm is one campaign e-mail.",
        "Inherited from Part I.",
        "Only through the data documentation; delivery and opening were not recorded.",
    ),
    "No interference (SUTVA)": (
        "fundamentally untestable",
        "Assumed: one customer's e-mail does not change another's spending.",
        "Inherited from Part I.",
        "No household or network identifiers exist to examine it.",
    ),
    "Temporal ordering": (
        "supported by design",
        "Covariates describe the prior twelve months; outcomes the two weeks after the campaign.",
        "Same.",
        "Documented by the data source.",
    ),
    "Outcome and propensity model specification": (
        "partially testable",
        "Not needed for unbiasedness; regression adjustment only sharpens precision.",
        "Matters for every adjusted estimator.",
        "Alternative specifications, doubly robust estimators and the comparison with the benchmark.",
    ),
}

# Figures copied into docs/ so the README and the published report can show them on GitHub.
PUBLISHED_FIGURES = [
    "experiment/itt_effects.png",
    "benchmark/selection_imbalance.png",
    "propensity/estimated_propensity_overlap.png",
    "findings/estimator_recovery.png",
    "comparison/forest_display_draw_spend.png",
    "heterogeneity/gates_by_predicted_effect.png",
    "heterogeneity/uplift_curves_spend.png",
    "robustness/sensitivity_summary.png",
]

# (spec item from the project brief, figure path under reports/figures, figure function)
FIGURE_GUIDE = [
    ("Outcome distributions by arm", "experiment/outcome_levels_by_arm.png", viz.outcome_levels),
    ("Outcome distributions by arm", "experiment/spend_among_purchasers.png", viz.spend_among_converters),
    ("Covariate balance in the experiment", "experiment/balance_randomized.png", viz.love_plot),
    ("Experimental treatment effects", "experiment/itt_effects.png", viz.effect_forest),
    ("Imbalance after constructed selection", "benchmark/selection_imbalance.png", viz.selection_love_plot),
    ("Propensity overlap", "benchmark/design_propensity_overlap.png", viz.design_propensity_overlap),
    ("Propensity overlap", "propensity/estimated_propensity_overlap.png", viz.estimated_propensity_overlap),
    (
        "Love plot before and after adjustment",
        "propensity/balance_before_after_adjustment.png",
        viz.adjustment_love_plot,
    ),
    ("Observational estimator comparison", "comparison/forest_display_draw_spend.png", viz.comparison_forest),
    ("Bias relative to the experimental benchmark", "findings/estimator_recovery.png", viz.recovery_heatmap),
    ("Bias relative to the experimental benchmark", "comparison/bias_rmse_spend.png", viz.bias_rmse_summary),
    (
        "Bootstrap and sampling distributions",
        "experiment/bootstrap_spend_effects.png",
        viz.bootstrap_distributions,
    ),
    (
        "Bootstrap and sampling distributions",
        "comparison/sampling_distributions_spend.png",
        viz.estimator_error_grid,
    ),
    (
        "Distribution of heterogeneous effects",
        "heterogeneity/forest_cate_distribution.png",
        viz.cate_distribution,
    ),
    ("Effects by customer segment", "heterogeneity/gates_by_predicted_effect.png", viz.gates_plot),
    ("Effects by customer segment", "heterogeneity/segment_effects_spend.png", viz.segment_forest),
    ("Uplift curves", "heterogeneity/uplift_curves_spend.png", viz.uplift_curves),
    ("Sensitivity analysis", "robustness/sensitivity_summary.png", viz.sensitivity_summary),
    ("Sensitivity analysis", "robustness/specification_bias_spend.png", viz.specification_bias),
    ("Sensitivity analysis", "robustness/trimming_and_calipers.png", viz.trimming_and_calipers),
]


def load_results(paths: DataPaths) -> dict:
    """Saved tables from every analysis, after checking they all come from the same data and benchmark."""
    records = {name: read_provenance(paths.results(name)) for name in SOURCES}
    checksums = {record["data"]["processed_sha256"] for record in records.values()}
    if len(checksums) != 1:
        raise RuntimeError("results were produced from different processed datasets; rerun `make analysis`")
    check_provenance({name: r for name, r in records.items() if "benchmark" in r})

    def read(source: str, name: str) -> pd.DataFrame:
        return pd.read_csv(paths.results(source) / f"{name}.csv")

    return {
        "provenance": records,
        "summary": json.loads((paths.results("experiment") / "summary.json").read_text()),
        "effects": read("experiment", "effects"),
        "lift": read("experiment", "relative_lift"),
        "balance": read("experiment", "balance"),
        "decomposition": read("benchmark", "bias_decomposition"),
        "display": read("comparison", "display_draw"),
        "performance": read("comparison", "performance"),
        "verdicts": read("heterogeneity", "verdicts"),
        "policy_differences": read("heterogeneity", "policy_differences"),
        "experiment_robustness": read("robustness", "experiment_robustness"),
        "bootstrap_stability": read("robustness", "bootstrap_stability"),
        "refutation": read("robustness", "refutation_summary"),
        "sensitivity": read("robustness", "sensitivity_summary"),
    }


def _money(value: float) -> str:
    return f"-${-value:.2f}" if value < 0 else f"${value:.2f}"


def _effect(value: float, outcome: str) -> str:
    return _money(value) if outcome == "spend" else f"{value * 100:.2f} pp"


def _interval(low: float, high: float, outcome: str) -> str:
    return f"[{_money(low)}, {_money(high)}]" if outcome == "spend" else _fmt_ci(low, high, 100)


def _md(table: pd.DataFrame) -> str:
    cells = table.astype(str).apply(lambda col: col.str.replace("|", "\\|", regex=False))
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines)


def _sentence_list(items: list[str]) -> str:
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _figure(name: str, alt: str) -> str:
    return f"![{alt}](../../figures/{name})"


def experiment_table(effects: pd.DataFrame, lift: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in effects[effects["estimator"] == "difference_in_means"].itertuples(index=False):
        relative = lift[(lift["contrast"] == row.contrast) & (lift["outcome"] == row.outcome)].iloc[0]
        rows.append(
            {
                "Contrast": viz.CONTRAST_LABELS[row.contrast].split(" (")[0],
                "Role": row.role,
                "Outcome": row.outcome,
                "Estimand": "ITT effect of assignment",
                "Estimate": _effect(row.estimate, row.outcome),
                "95% CI": _interval(row.ci_low, row.ci_high, row.outcome),
                "Relative lift": f"{relative['relative_lift']:.0%}",
                "Holm p": _fmt_p(row.p_holm),
                "Randomization p": _fmt_p(row.p_randomization, floor=1e-4),
            }
        )
    return pd.DataFrame(rows)


def estimator_table(display: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    """One row per estimator, comparison, scenario and outcome: the comparison table from the project brief.

    The estimate and interval come from the fixed display draw; bias and coverage from all 200 draws.
    """
    keys = ["comparison", "scenario", "outcome", "estimator"]
    merged = display.merge(
        performance[[*keys, "bias_pct", "coverage", "unbiased_within_mc_error"]], on=keys, how="left"
    )
    merged = merged[merged["estimator"].isin(TABLE_ESTIMATORS) & merged["outcome"].isin(["spend", "visit"])]
    order = {name: i for i, name in enumerate(TABLE_ESTIMATORS)}
    merged = merged.sort_values(keys, key=lambda col: col.map(order) if col.name == "estimator" else col)
    rows = []
    for row in merged.itertuples(index=False):
        experimental = row.estimator.startswith("experiment")
        if experimental:
            bias = "benchmark"
        else:
            bias = f"{row.bias_pct:+.1%}" + ("" if row.unbiased_within_mc_error else " *")
        rows.append(
            {
                "comparison": row.comparison,
                "scenario": row.scenario,
                "outcome": row.outcome,
                "Estimator": viz.ESTIMATOR_DISPLAY.get(row.estimator, "Randomized experiment"),
                "Estimand": row.estimand,
                "Estimated effect (draw 0)": _effect(row.estimate, row.outcome),
                "95% CI (draw 0)": _interval(row.ci_low, row.ci_high, row.outcome),
                "Mean bias vs benchmark (200 draws)": bias,
                "CI coverage": "—" if experimental or np.isnan(row.coverage) else f"{row.coverage:.0%}",
                "Key assumptions": row.key_assumptions,
            }
        )
    return pd.DataFrame(rows)


def _recovering(performance: pd.DataFrame, outcome: str, scenario: str) -> list[str]:
    """Feasible estimators whose bias is within Monte Carlo error for both comparisons."""
    cell = performance[(performance["outcome"] == outcome) & (performance["scenario"] == scenario)]
    clean = cell.groupby("estimator")["unbiased_within_mc_error"].all()
    feasible = [e for e in RECOVERY_ESTIMATORS if e not in ("naive_difference", "oracle_ipw")]
    return [viz.ESTIMATOR_DISPLAY[e] for e in feasible if clean.get(e, False)]


def _part_one(r: dict, tables: dict) -> list[str]:
    dim = r["effects"][r["effects"]["estimator"] == "difference_in_means"].set_index(["contrast", "outcome"])
    lift = r["lift"].set_index(["contrast", "outcome"])
    diagnostics = pd.DataFrame(r["summary"]["diagnostics"]).set_index("contrast")
    plan = r["summary"]["plan"]
    rows = next(iter(r["provenance"].values()))["data"]["rows"]

    def effect(contrast: str, outcome: str = "spend") -> str:
        row = dim.loc[(contrast, outcome)]
        interval = _interval(row["ci_low"], row["ci_high"], outcome)
        return f"{_effect(row['estimate'], outcome)} (95% CI {interval})"

    versions = r["experiment_robustness"]
    spend = versions[(versions["analysis"] == "spend version") & ~versions["outcome"].str.startswith("log")]
    dedup = versions[
        versions["analysis"].str.startswith("without duplicate") & (versions["outcome"] == "spend")
    ]
    dedup = dedup.set_index("contrast")["estimate"]
    lower = r["bootstrap_stability"].groupby("contrast")["ci_low"]

    def spend_range(contrast: str) -> str:
        values = spend.loc[spend["contrast"] == contrast, "estimate"]
        return f"{_money(values.min())} to {_money(values.max())}"

    control = lift.loc[("mens_vs_control", "spend")]
    versus_p = dim.loc[("mens_vs_womens", "spend"), "p_randomization"]
    return [
        "## 1. What randomization establishes",
        "",
        f"The Hillstrom (2008) experiment randomly assigned {rows:,} customers to a Mens e-mail, a Womens "
        "e-mail or no e-mail. Because assignment was random, the difference in means is an unbiased "
        "estimate of the effect of assignment, with no modelling assumptions.",
        "",
        _md(tables["experiment_table"]),
        "",
        "Holm p-values adjust the two co-primary contrasts for multiplicity; randomization p-values come "
        f"from {plan['inference']['permutations']:,} re-randomizations. Visit and conversion effects are in "
        "percentage points.",
        "",
        f"- Assignment to the **Mens e-mail** raised mean two-week spend by {effect('mens_vs_control')}, "
        f"a {control['relative_lift']:.0%} lift over the No E-Mail mean of "
        f"{_money(control['control_mean'])}.",
        f"- Assignment to the **Womens e-mail** raised it by {effect('womens_vs_control')}.",
        f"- The Mens e-mail out-earned the Womens e-mail by {effect('mens_vs_womens')} (secondary "
        f"contrast, randomization p = {_fmt_p(versus_p, 1e-4)}).",
        f"- Visit rates rose by {effect('mens_vs_control', 'visit')} (Mens) and "
        f"{effect('womens_vs_control', 'visit')} (Womens).",
        "",
        "**Is the randomization credible?** A classifier cannot tell the arms apart from pre-treatment "
        "covariates (out-of-fold AUC "
        + " and ".join(f"{diagnostics.loc[c, 'out_of_fold_auc']:.3f}" for c in CONTRASTS)
        + ", likelihood-ratio p = "
        + " and ".join(f"{diagnostics.loc[c, 'lr_p_value']:.2f}" for c in CONTRASTS)
        + ", Mens and Womens), and the largest absolute standardized mean difference is "
        f"{r['balance']['smd'].abs().max():.3f}.",
        "",
        "**Is the spend effect robust?** Across the pre-specified spend versions (recoding the $499 values, "
        f"winsorizing) the Mens estimate ranges from {spend_range('mens_vs_control')} and the Womens "
        f"estimate from {spend_range('womens_vs_control')}. Keeping one copy of each exact duplicate row "
        f"gives {_money(dedup['mens_vs_control'])} and {_money(dedup['womens_vs_control'])}. Under five "
        "further bootstrap seeds the lower bound of the Mens BCa interval stays between "
        f"${lower.min()['mens_vs_control']:.3f} and ${lower.max()['mens_vs_control']:.3f}.",
        "",
        _figure("experiment/itt_effects.png", "ITT effects with 95% intervals"),
        "",
    ]


def _part_two(r: dict, tables: dict) -> list[str]:
    perf = r["performance"]
    naive = r["decomposition"]
    naive = naive[(naive["scenario"] == "A") & (naive["outcome"] == "spend")].set_index("comparison")
    lines = [
        "## 2. What happens without randomization",
        "",
        "Part II removes the protection of randomization on purpose. Customers are subsampled from each "
        "randomized pair with a known selection rule (RCT rejection sampling, Keith et al. 2023), so that "
        "high-spending, recent, multichannel and established customers are more likely to appear e-mailed. "
        "The full randomized pair still supplies the true answer. Three scenarios, 200 draws each:",
        "",
        "- **A**: moderate selection on covariates the analyst observes.",
        "- **B**: twice as strong, which stresses overlap and produces extreme weights.",
        "- **C**: the strength of A, but recency is hidden from the analyst, so unconfoundedness fails.",
        "",
        "In Scenario A the naive comparison overstates the spend effect by "
        + " and ".join(
            f"{naive.loc[c, 'bias_vs_ate'] / naive.loc[c, 'ate']:.0%} ({CONTRASTS[c]})" for c in CONTRASTS
        )
        + ". The error splits exactly into a selection term (the e-mailed customers would have spent more "
        "anyway) and an effect-heterogeneity term (ATT minus ATE, which can take either sign): "
        + "; ".join(
            f"{_money(naive.loc[c, 'baseline_bias'])} and {_money(naive.loc[c, 'heterogeneity_term'])} "
            f"per customer for {CONTRASTS[c]}"
            for c in CONTRASTS
        )
        + ".",
        "",
        "## 3. Which estimators recover the experimental answer",
        "",
        _figure("findings/estimator_recovery.png", "Mean bias of each estimator by comparison and scenario"),
        "",
        "Feasible estimators whose mean bias is within Monte Carlo error for both comparisons:",
        "",
    ]
    for scenario, label in (("A", "moderate selection"), ("B", "strong selection"), ("C", "recency hidden")):
        lines.append(
            f"- **Scenario {scenario}** ({label}): "
            f"spend: {_sentence_list(_recovering(perf, 'spend', scenario))}; "
            f"visit: {_sentence_list(_recovering(perf, 'visit', scenario))}."
        )
    cells = perf[
        ~perf["estimator"].isin(["naive_difference", "oracle_ipw"])
        & perf["outcome"].isin(["spend", "visit"])
        & perf["scenario"].isin(["A", "B"])
    ]
    lines += [
        "",
        "Each flag is a separate test at roughly the 5% level, so among the "
        f"{len(cells)} feasible cells in Scenarios A and B about {0.05 * len(cells):.0f} would be flagged by "
        "chance even if every estimator were unbiased. An isolated flag, such as one comparison in one "
        "scenario, is weak evidence on its own; a pattern across cells is not.",
        "",
    ]
    matching = perf[(perf["estimator"] == "matching_nn") & perf["outcome"].isin(["spend", "visit"])]
    if not matching["unbiased_within_mc_error"].any():
        lines += [
            "1:1 matching with a 0.2-SD caliper is biased in every cell: the caliper discards treated "
            "customers without a close control, which changes who the estimate describes.",
            "",
        ]
    if not _recovering(perf, "spend", "C") and not _recovering(perf, "visit", "C"):
        lines += [
            "With recency hidden (C), no feasible estimator recovers the benchmark: adjustment cannot remove "
            "bias from a variable the analyst does not observe.",
            "",
        ]
    lines += [
        "The tables below are the estimator comparison for spend in Scenario A. The estimate and interval "
        "come from one constructed dataset (draw 0, fixed before any result was seen); bias and coverage "
        "summarize all 200 draws. `*` marks a bias larger than twice its Monte Carlo standard error. Every "
        "scenario and outcome is in `estimator_table.csv`.",
        "",
    ]
    table = tables["estimator_table"]
    for comparison, label in CONTRASTS.items():
        shown = table[
            (table["comparison"] == comparison) & (table["scenario"] == "A") & (table["outcome"] == "spend")
        ]
        lines += [
            f"**{label}, Scenario A, spend**",
            "",
            _md(shown.drop(columns=["comparison", "scenario", "outcome"])),
            "",
        ]
    lines += [
        "ATE and ATT estimators are compared only with the benchmark for their own estimand. The naive "
        "difference targets no causal estimand; it is set against the ATE only to show its error.",
        "",
        _figure("comparison/forest_display_draw_spend.png", "Estimates from one constructed dataset"),
        "",
    ]
    return lines


def _heterogeneity(r: dict) -> list[str]:
    verdicts = r["verdicts"]
    detected = verdicts[verdicts["heterogeneity_detected"]]
    differences = r["policy_differences"]
    best = differences[
        (differences["policy"] == "model_best_arm")
        & (differences["versus"] == "treat_all_mens")
        & (differences["outcome"] == "spend")
    ].iloc[0]
    table = pd.DataFrame(
        {
            "Contrast": verdicts["contrast"].map(CONTRASTS),
            "Outcome": verdicts["outcome"],
            "BLP slope b2": verdicts["b2"].map("{:.2f}".format),
            "p (split-adjusted)": verdicts["b2_p_value"].map(_fmt_p),
            "Heterogeneity detected": verdicts["heterogeneity_detected"].map({True: "yes", False: "no"}),
        }
    )
    if len(detected):
        found = _sentence_list([f"{CONTRASTS[d.contrast]} on {d.outcome}" for d in detected.itertuples()])
        verdict = (
            f"Heterogeneity was detected only for {found}. Elsewhere the predicted differences did not "
            "replicate on held-out data."
        )
    else:
        verdict = (
            "No heterogeneity was detected: the predicted differences did not replicate on held-out data."
        )
    # The bootstrap resamples customers with the fitted predictions held fixed, so it leaves out
    # model-fitting variability and the interval is somewhat too narrow.
    if best["ci_low"] <= 0:
        policy = (
            "The experiment therefore supports the simple policy over any targeting these covariates allow. "
            "The interval holds the fitted predictions fixed, so it leaves out model-fitting variability; "
            "a wider interval would only strengthen this conclusion."
        )
    else:
        policy = (
            "Model-based targeting beats sending everyone the Mens e-mail, although the interval holds the "
            "fitted predictions fixed and so understates the uncertainty."
        )
    return [
        "## 4. Does the effect differ between customers?",
        "",
        "Heterogeneity was claimed only if the pre-specified tests (the BLP slope b2, or the spread of "
        "sorted group effects, over 100 sample splits) supported it.",
        "",
        _md(table),
        "",
        verdict,
        "",
        "**Targeting.** Sending each customer the e-mail the model predicts is best changes expected spend "
        f"by {_money(best['difference'])} per customer, compared with sending everyone the Mens e-mail "
        f"(95% CI {_interval(best['ci_low'], best['ci_high'], 'spend')}). {policy} These are estimated "
        "values of policies that were never run, not realized outcomes.",
        "",
        _figure("heterogeneity/gates_by_predicted_effect.png", "Sorted group average effects"),
        "",
    ]


def _robustness(r: dict) -> list[str]:
    refutation = r["refutation"]
    placebo = refutation.groupby("scenario")["placebo_rejection_rate"].mean()
    bias_c = refutation.loc[
        (refutation["scenario"] == "C") & (refutation["outcome"] == "spend"), "original_bias_pct"
    ]
    sens = r["sensitivity"].set_index(["comparison", "scenario"])
    rv = {c: (sens.loc[(c, "A"), "visit_rv_q1"], sens.loc[(c, "C"), "visit_rv_q1"]) for c in CONTRASTS}
    reading = (
        ": the estimates look more robust when a confounder is hidden, not less. "
        if all(c > a for a, c in rv.values())
        else ". "
    )
    covered = _sentence_list(
        [f"{sens.loc[('mens_vs_control', 'C'), f'visit_bound_k{k}_covers_error']:.0%}" for k in (1, 2, 3)]
    )
    return [
        "## 5. Robustness and sensitivity",
        "",
        f"**Refutation tests do not detect hidden confounding.** A placebo treatment is rejected in "
        f"{placebo['A']:.0%} (A) and {placebo['C']:.0%} (C) of draws on average, and adding a random "
        "covariate or refitting on 80% subsets barely moves the estimates, in Scenario C just as in A. Yet "
        f"in C the spend estimates are biased by {bias_c.abs().min():.0%} to {bias_c.abs().max():.0%}. "
        "Passing a refutation test is not evidence that no confounder is missing.",
        "",
        "**Sensitivity analysis bounds; it does not detect.** The robustness value for visit is "
        + " and ".join(f"{a:.3f} (A) vs {c:.3f} (C) for {CONTRASTS[k]}" for k, (a, c) in rv.items())
        + reading
        + "What sensitivity analysis does do is turn a hypothesis about confounder strength into a bound. "
        "Bounds for a confounder one, two or three times as strong as the observed covariate giving the "
        f"largest bound cover the true visit error in {covered} of Mens draws in Scenario C.",
        "",
        _figure("robustness/sensitivity_summary.png", "Sensitivity summaries in Scenarios A and C"),
        "",
    ]


def _uncertainty(r: dict) -> list[str]:
    perf = r["performance"]

    def coverage(estimator: str, outcome: str) -> str:
        cells = perf[
            (perf["scenario"] == "C") & (perf["estimator"] == estimator) & (perf["outcome"] == outcome)
        ]
        cells = cells.set_index("comparison")["coverage"]
        return " and ".join(f"{cells[c]:.0%}" for c in CONTRASTS)

    return [
        "## 6. Statistical uncertainty and identification uncertainty",
        "",
        "Confidence intervals describe sampling variability under the stated assumptions. They say nothing "
        "about whether the assumptions hold.",
        "",
        "- **Part I** intervals are the whole story: randomization secures identification, so the remaining "
        "uncertainty is statistical.",
        "- **Part II, Scenario C**: with recency hidden, the cross-fitted AIPW interval for the visit effect "
        f"covers the truth in {coverage('aipw', 'visit')} of draws (Mens, Womens), against "
        f"{coverage('oracle_ipw', 'visit')} for the oracle. The intervals are centred in the wrong place, "
        "and no standard error can show that. For spend the same intervals still cover in "
        f"{coverage('aipw', 'spend')} of draws, only because spend is noisy enough to make them wide: high "
        "coverage there is not evidence of small bias.",
        "",
    ]


def _conclusions(r: dict, tables: dict) -> list[str]:
    dim = r["effects"][r["effects"]["estimator"] == "difference_in_means"].set_index("contrast")
    increased = bool((dim.loc[list(CONTRASTS), "ci_low"] > 0).all())
    mens_ahead = bool((dim.loc["mens_vs_womens", "ci_low"] > 0).all())
    detected = r["verdicts"][r["verdicts"]["heterogeneity_detected"]]
    if increased:
        effects = "- Assignment to either e-mail increased two-week spend, visits and conversions"
        effects += ", and the Mens e-mail outperformed the Womens e-mail on all three." if mens_ahead else "."
    else:
        effects = "- The direction and size of each effect in the table in Section 1."
    exceptions = _sentence_list(
        [f"{d.outcome} under the {CONTRASTS[d.contrast]} contrast" for d in detected.itertuples()]
    )
    return [
        "## 7. Assumptions",
        "",
        _md(tables["assumptions"]),
        "",
        "## 8. What the evidence supports, and what it does not",
        "",
        "**Supported by the randomized design alone:**",
        "",
        effects,
        "- The size of those effects, with the stated statistical uncertainty.",
        "- Little evidence that the effect varies with these covariates"
        + (f", except {exceptions}." if len(detected) else "."),
        "",
        "**Requires further assumptions:**",
        "",
        "- Any effect of *receiving* or *opening* an e-mail. Neither was recorded; estimating it would need "
        "an instrumental-variable analysis and its exclusion restriction.",
        "- Transport to other retailers, years or campaigns. The data are from 2008 and the outcome window "
        "is two weeks.",
        "- Any conclusion from real observational data. Part II shows what estimators recover when the "
        "assumptions hold or fail by construction; it cannot certify those assumptions where the selection "
        "rule is unknown.",
        "",
        "**Where the conclusions stop.** Subgroup estimates are imprecise and exploratory. The observational "
        "scenarios are constructed, with a selection rule chosen in advance; real selection may be stronger, "
        "nonlinear or driven by unrecorded variables. The data hold an observed maximum of $499; whether it "
        "reflects a true maximum or a data-generation constraint is unknown, and the spend effect barely "
        "moves when those values are recoded or winsorized.",
        "",
    ]


def _provenance(r: dict) -> list[str]:
    records = r["provenance"]
    first = next(iter(records.values()))
    benchmark = records["comparison"]["benchmark"]
    commits = ", ".join(f"`{c[:7]}`" for c in sorted({rec["code"]["commit"] for rec in records.values()}))
    dirty = any(rec["code"]["uncommitted_changes"] for rec in records.values())
    packages = ", ".join(f"{k} {v}" for k, v in first["environment"]["packages"].items())
    return [
        "## 10. Provenance",
        "",
        f"- Processed data SHA-256 `{first['data']['processed_sha256']}` ({first['data']['rows']:,} rows, "
        f"none removed); raw data SHA-256 `{first['data']['raw_sha256']}`.",
        f"- Benchmark specification v{benchmark['version']} (SHA-256 `{benchmark['sha256']}`), base seed "
        f"{benchmark['base_seed']}, {benchmark['n_draws']} draws.",
        f"- Code {commits}" + (", with uncommitted changes to `src/` or `configs/`." if dirty else "."),
        f"- Python {first['environment']['python']}; {packages}.",
        "",
    ]


def findings_markdown(results: dict, tables: dict[str, pd.DataFrame]) -> str:
    estimand = results["summary"]["plan"]["meta"]["estimand"]
    lines = [
        "# Causeway: findings",
        "",
        "Does assigning a customer to a marketing e-mail make them spend more, or are the customers who get "
        "promotions simply the ones who would have bought anyway? This report gathers the answers from every "
        "stage of the analysis. Every number is read from the saved result tables.",
        "",
        f"Estimand throughout Part I: the **{estimand}**, over the two weeks after the campaign. It is the "
        "effect of being assigned an e-mail, not of receiving or opening one.",
        "",
        *_part_one(results, tables),
        *_part_two(results, tables),
        *_heterogeneity(results),
        *_robustness(results),
        *_uncertainty(results),
        *_conclusions(results, tables),
        "## 9. Figure guide",
        "",
        "Each figure answers one question, taken from the docstring of the function that draws it.",
        "",
        _md(tables["figure_guide"]),
        "",
        *_provenance(results),
    ]
    return "\n".join(lines)


def build_findings_report(paths: DataPaths | None = None) -> ReportOutputs:
    paths = paths or DataPaths()
    results = load_results(paths)
    results_dir, figures_dir = paths.results("findings"), paths.figures("findings")
    tables = {
        "experiment_table": experiment_table(results["effects"], results["lift"]),
        "estimator_table": estimator_table(results["display"], results["performance"]),
        "assumptions": pd.DataFrame(
            [(name, *values) for name, values in ASSUMPTIONS.items()],
            columns=["Assumption", "Status", "Part I", "Part II", "How it is examined"],
        ),
        "figure_guide": pd.DataFrame(
            [
                (item, f"[{path}](../../figures/{path})", function.__doc__.strip())
                for item, path, function in FIGURE_GUIDE
            ],
            columns=["Required visualization", "Figure", "Question it answers"],
        ),
    }
    _save_tables(results_dir, tables)
    figures = [
        viz.recovery_heatmap(
            results["performance"], figures_dir / "estimator_recovery.png", RECOVERY_ESTIMATORS
        )
    ]
    missing = [path for _, path, _ in FIGURE_GUIDE if not paths.figures(path).exists()]
    if missing:
        raise FileNotFoundError(f"figures missing, rerun `make analysis`: {', '.join(missing)}")
    report_path = results_dir / "findings_report.md"
    report_path.write_text(findings_markdown(results, tables))
    return ReportOutputs(tables, figures, report_path)


def relink(markdown: str) -> str:
    """Point figure links at docs/figures; figures that are not published become plain paths."""

    def replace(match: re.Match) -> str:
        bang, text, path = match.groups()
        if path in PUBLISHED_FIGURES:
            return f"{bang}[{text}](figures/{Path(path).name})"
        return f"`reports/figures/{path}`"

    return re.sub(r"(!?)\[([^\]]*)\]\(\.\./\.\./figures/([^)]+)\)", replace, markdown)


def publish_docs(paths: DataPaths | None = None, docs: Path = PROJECT_ROOT / "docs") -> list[Path]:
    """Copy the findings report and the published figures into docs/ for the README to link to."""
    paths = paths or DataPaths()
    report = paths.results("findings") / "findings_report.md"
    if not report.exists():
        raise FileNotFoundError(f"{report} not found; run `make analysis` first")
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    written = []
    for path in PUBLISHED_FIGURES:
        target = docs / "figures" / Path(path).name
        shutil.copyfile(paths.figures(path), target)
        written.append(target)
    note = (
        "> Copied from `reports/results/findings/findings_report.md` by `make docs`. Figures marked as paths "
        "and every result table are produced locally by `make analysis`.\n\n"
    )
    title, _, body = report.read_text().partition("\n")
    target = docs / "findings.md"
    target.write_text(f"{title}\n\n{note}{relink(body.lstrip())}")
    return [target, *written]
