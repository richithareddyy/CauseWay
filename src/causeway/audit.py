"""Check every number in README.md and docs/findings.md against committed files.

Needs no raw data and refits nothing.

Each sentence or table row that quotes a number is rebuilt here from the committed tables and must appear in
the document exactly. Any number left unexplained is a failure. The formatting is written separately from
the code that generates the report, so a mistake there would not be repeated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = ["experiment", "benchmark", "propensity", "estimation", "comparison", "heterogeneity", "robustness"]
TABLES = {
    "effects": "reports/results/experiment/effects.csv",
    "lift": "reports/results/experiment/relative_lift.csv",
    "balance": "reports/results/experiment/balance.csv",
    "diagnostics": "reports/results/experiment/diagnostics.csv",
    "decomposition": "reports/results/benchmark/bias_decomposition.csv",
    "display": "reports/results/comparison/display_draw.csv",
    "performance": "reports/results/comparison/performance.csv",
    "verdicts": "reports/results/heterogeneity/verdicts.csv",
    "policy": "reports/results/heterogeneity/policy_differences.csv",
    "robustness": "reports/results/robustness/experiment_robustness.csv",
    "stability": "reports/results/robustness/bootstrap_stability.csv",
    "refutation": "reports/results/robustness/refutation_summary.csv",
    "sensitivity": "reports/results/robustness/sensitivity_summary.csv",
    "estimator_table": "reports/results/findings/estimator_table.csv",
}
VALIDATION = "reports/validation/validation_raw.json"
COMMITTED = [*TABLES.values(), VALIDATION, *(f"reports/results/{s}/provenance.json" for s in SOURCES)]
MAX_BYTES, MAX_ROWS = 5_000_000, 500

PAIRS = {"mens_vs_control": "Mens vs No E-Mail", "womens_vs_control": "Womens vs No E-Mail"}
CONTRASTS = {**PAIRS, "mens_vs_womens": "Mens vs Womens"}
OUTCOMES = ["spend", "visit", "conversion"]
ADJUSTED = ["outcome_regression", "ipw_ate", "aipw", "dml_irm", "ipw_att", "matching_nn"]
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 8: "eight"}
NUMBER = re.compile(
    r"\d|\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twice|third|half)\b", re.IGNORECASE
)
EFFECTS = "experiment/effects.csv"
PERFORMANCE = "comparison/performance.csv"
FROZEN = "configs/benchmark/v1.toml"
VERDICTS = "heterogeneity/verdicts.csv"
SENSITIVITY = "robustness/sensitivity_summary.csv"
DECOMPOSITION = "benchmark/bias_decomposition.csv"
SECTIONS = [
    "What randomization establishes",
    "What happens without randomization",
    "Which estimators recover the experimental answer",
    "Does the effect differ between customers?",
    "Robustness and sensitivity",
    "Statistical uncertainty and identification uncertainty",
    "Assumptions",
    "What the evidence supports, and what it does not",
    "Figure guide",
    "Provenance",
]


class AuditError(RuntimeError):
    """A quoted number, or the evidence behind it, does not check out."""


@dataclass(frozen=True)
class Claim:
    text: str
    source: str
    times: int = 1


@dataclass
class Evidence:
    tables: dict[str, pd.DataFrame]
    provenance: dict[str, dict]
    validation: dict
    configs: dict[str, dict]
    design: str
    pyproject: dict


def require(name: str, passed: bool, detail: str = "") -> None:
    if not passed:
        raise AuditError(f"{name}{': ' + detail if detail else ''}")


# Formatting, deliberately not shared with causeway.findings


def money(value: float) -> str:
    return f"{'-' if value < 0 else ''}${abs(value):.2f}"


def amount(value: float, outcome: str) -> str:
    return money(value) if outcome == "spend" else f"{100 * value:.2f} pp"


def interval(low: float, high: float, outcome: str) -> str:
    if outcome == "spend":
        return f"[{money(low)}, {money(high)}]"
    return f"[{100 * low:.2f}, {100 * high:.2f}]"


def percent(value: float) -> str:
    return f"{value:.0%}"


def p_value(value: float, smallest: float | None = None) -> str:
    if np.isnan(value):
        return "—"
    if smallest is not None and value <= smallest:
        return f"≤ {smallest:.4f}"
    return "< 0.0001" if value < 0.0001 else f"{value:.4f}"


def listing(items: list[str]) -> str:
    if len(items) < 2:
        return items[0] if items else "none"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


# Evidence


def load(root: Path) -> Evidence:
    for name in COMMITTED:
        path = root / name
        require("committed file missing", path.is_file(), name)
        require("committed file too large", path.stat().st_size < MAX_BYTES, name)
    tables = {name: pd.read_csv(root / path) for name, path in TABLES.items()}
    for name, table in tables.items():
        require("committed table is not an aggregate", len(table) <= MAX_ROWS, TABLES[name])
    names = ["data", "experiment", "heterogeneity", "robustness"]
    configs = {name: tomllib.loads((root / f"configs/{name}.toml").read_text()) for name in names}
    configs["benchmark"] = tomllib.loads((root / FROZEN).read_text())
    configs["criteo"] = tomllib.loads((root / "configs/extension/criteo_gate.toml").read_text())
    return Evidence(
        tables,
        {s: json.loads((root / f"reports/results/{s}/provenance.json").read_text()) for s in SOURCES},
        json.loads((root / VALIDATION).read_text()),
        configs,
        (root / "docs/design.md").read_text(),
        tomllib.loads((root / "pyproject.toml").read_text()),
    )


def display_names(e: Evidence) -> dict[str, str]:
    ratio = e.configs["benchmark"]["estimators"]
    ratio = next(x for x in ratio if x["name"] == "matching_nn")["matching"]["ratio"]
    return {
        "experiment": "Randomized experiment",
        "experiment_att": "Randomized experiment",
        "naive_difference": "Naive difference",
        "oracle_ipw": "Oracle IPW (true e*)",
        "outcome_regression": "Outcome regression (OLS)",
        "ipw_ate": "IPW (ATE)",
        "aipw": "AIPW (linear, cross-fitted)",
        "dml_irm": "DML (gradient boosting)",
        "ipw_att": "IPW (ATT)",
        "matching_nn": f"1:{ratio} NN matching (ATT)",
    }


def check_provenance(root: Path, e: Evidence) -> None:
    first = e.provenance[SOURCES[0]]
    benchmark = e.configs["benchmark"]
    digest = hashlib.sha256((root / FROZEN).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    for source, record in e.provenance.items():
        require("provenance data", record["data"] == first["data"], source)
        require("provenance code", record["code"] == first["code"], source)
        require("provenance environment", record["environment"] == first["environment"], source)
        require("provenance clean tree", record["code"]["uncommitted_changes"] is False, source)
        if "benchmark" in record:
            expected = {
                "version": benchmark["meta"]["benchmark_version"],
                "sha256": digest,
                "base_seed": benchmark["randomness"]["base_seed"],
                "n_draws": benchmark["randomness"]["n_draws"],
            }
            require("provenance benchmark configuration", record["benchmark"] == expected, source)
    stamped = [s for s in SOURCES if "benchmark" in e.provenance[s]]
    require("benchmark configuration recorded", len(stamped) == 5, str(stamped))
    data = e.configs["data"]
    require("provenance raw checksum", first["data"]["raw_sha256"] in {s["sha256"] for s in data["sources"]})
    require("provenance rows", first["data"]["rows"] == data["dataset"]["expected_rows"])
    for name in ("experiment", "heterogeneity"):
        recorded = e.provenance[name][f"{name}_plan_version"]
        require("provenance plan version", recorded == e.configs[name]["meta"]["version"], name)


def check_tables(e: Evidence) -> None:
    """The committed tables must agree with each other and with the configuration."""
    t = e.tables
    effects, perf, display = t["effects"], t["performance"], t["display"]
    dim = effects[effects["estimator"] == "difference_in_means"].set_index(["contrast", "outcome"])
    for outcome in OUTCOMES:
        small, large = sorted((dim.loc[(c, outcome), "p_value"], c) for c in PAIRS)
        holm = {small[1]: min(1.0, 2 * small[0])}
        holm[large[1]] = max(holm[small[1]], min(1.0, large[0]))
        for contrast, value in holm.items():
            saved = dim.loc[(contrast, outcome), "p_holm"]
            require("Holm adjustment", np.isclose(saved, value, rtol=1e-9), f"{contrast} {outcome}")
    lift = t["lift"].set_index(["contrast", "outcome"])
    ratio = dim["estimate"] / lift["control_mean"].reindex(dim.index)
    require("relative lift", np.allclose(ratio, lift["relative_lift"].reindex(dim.index)))

    require("draws per cell", set(perf["n_draws"]) == {e.configs["benchmark"]["randomness"]["n_draws"]})
    require("bias share", np.allclose(perf["bias"] / perf["benchmark_value"], perf["bias_pct"]))
    # Not quoted anywhere, but tied to the quoted columns by definition.
    n, sd = perf["n_draws"], perf["empirical_sd"]
    require("no failed draws", (perf["failure_rate"] == 0).all())
    require("mean estimate", np.allclose(perf["mean_estimate"] - perf["benchmark_value"], perf["bias"]))
    require("Monte Carlo SE", np.allclose(sd / np.sqrt(n), perf["bias_mc_se"]))
    require("RMSE", np.allclose(np.sqrt(perf["bias"] ** 2 + sd**2 * (n - 1) / n), perf["rmse"]))
    require("SE to SD ratio", np.allclose(perf["mean_se"] / sd, perf["se_to_sd_ratio"], rtol=1e-6))
    gap = perf["coverage"] - perf["oracle_coverage"]
    require("coverage gap", np.allclose(gap, perf["coverage_minus_oracle"], equal_nan=True))
    flag = perf["bias"].abs() <= 2 * perf["bias_mc_se"]
    require("bias flag", (flag == perf["unbiased_within_mc_error"]).all())
    d = t["decomposition"]
    require("bias decomposition", np.allclose(d["naive_target"] - d["ate"], d["bias_vs_ate"]))
    require("bias decomposition", np.allclose(d["att"] - d["ate"], d["heterogeneity_term"]))
    require(
        "bias split is exact", np.allclose(d["baseline_bias"] + d["heterogeneity_term"], d["bias_vs_ate"])
    )
    for r in d.itertuples():
        require("ATE reference", np.isclose(r.ate, dim.loc[(r.comparison, r.outcome), "estimate"]))
        cell = perf[
            (perf["comparison"] == r.comparison)
            & (perf["scenario"] == r.scenario)
            & (perf["outcome"] == r.outcome)
        ]
        for estimand, value in (("ATE", r.ate), ("ATT", r.att)):
            saved = cell.loc[cell["benchmark"] == estimand, "benchmark_value"]
            require("benchmark value", np.allclose(saved, value), f"{r.comparison} {r.scenario} {r.outcome}")
    for r in display[display["estimator"] == "experiment"].itertuples():
        saved = dim.loc[(r.comparison, r.outcome)]
        require(
            "experiment row",
            np.allclose(
                [r.estimate, r.ci_low, r.ci_high], saved[["estimate", "ci_low", "ci_high"]].astype(float)
            ),
        )


def estimator_rows(e: Evidence, comparison: str, scenario: str, outcome: str) -> list[list[str]]:
    display, perf = e.tables["display"], e.tables["performance"]

    def pick(table: pd.DataFrame) -> pd.DataFrame:
        match = (
            (table["comparison"] == comparison)
            & (table["scenario"] == scenario)
            & (table["outcome"] == outcome)
        )
        return table[match].set_index("estimator")

    shown, scored, rows = pick(display), pick(perf), []
    for estimator, label in display_names(e).items():
        d = shown.loc[estimator]
        bias, coverage = "benchmark", "—"
        if estimator in scored.index:
            s = scored.loc[estimator]
            beyond = abs(s["bias"]) > 2 * s["bias_mc_se"]
            bias = f"{s['bias'] / s['benchmark_value']:+.1%}{' *' if beyond else ''}"
            coverage = percent(s["coverage"])
        estimate, bounds = amount(d["estimate"], outcome), interval(d["ci_low"], d["ci_high"], outcome)
        rows.append([label, d["estimand"], estimate, bounds, bias, coverage, d["key_assumptions"]])
    return rows


def check_estimator_table(e: Evidence) -> None:
    saved = e.tables["estimator_table"]
    perf = e.tables["performance"]
    cells = saved[["comparison", "scenario", "outcome"]].drop_duplicates()
    require(
        "estimator table cells", len(cells) == perf["comparison"].nunique() * perf["scenario"].nunique() * 2
    )
    for comparison, scenario, outcome in cells.itertuples(index=False):
        match = (
            (saved["comparison"] == comparison)
            & (saved["scenario"] == scenario)
            & (saved["outcome"] == outcome)
        )
        rows = saved[match].iloc[:, 3:].astype(str).to_numpy().tolist()
        require(
            "estimator table",
            rows == estimator_rows(e, comparison, scenario, outcome),
            f"{comparison} {scenario} {outcome}",
        )


# Claims


def findings_claims(e: Evidence) -> list[Claim]:
    t, config = e.tables, e.configs
    effects, perf = t["effects"], t["performance"]
    names = display_names(e)
    benchmark, plan = config["benchmark"], config["experiment"]["inference"]
    draws, draw = benchmark["randomness"]["n_draws"], benchmark["randomness"]["fixed_display_draw"]
    level = f"{plan['ci_level']:.0%}"
    data = e.provenance["experiment"]["data"]
    dim = effects[effects["estimator"] == "difference_in_means"].set_index(["contrast", "outcome"])
    lift = t["lift"].set_index(["contrast", "outcome"])
    smallest = 1 / (plan["permutations"] + 1)
    smallest = math.ceil(smallest * 1e4) / 1e4
    claims: list[Claim] = []

    def add(text: str, source: str, times: int = 1) -> None:
        claims.append(Claim(text, source, times))

    def effect(contrast: str, outcome: str = "spend") -> str:
        r = dim.loc[(contrast, outcome)]
        return f"{amount(r['estimate'], outcome)} ({level} CI {interval(r['ci_low'], r['ci_high'], outcome)})"

    def scored(comparison: str, scenario: str, estimator: str, outcome: str) -> pd.Series:
        match = (perf["comparison"] == comparison) & (perf["scenario"] == scenario)
        return perf[match & (perf["estimator"] == estimator) & (perf["outcome"] == outcome)].iloc[0]

    def unbiased(estimator: str, scenario: str, outcome: str) -> bool:
        cells = [scored(c, scenario, estimator, outcome) for c in PAIRS]
        return all(abs(c["bias"]) <= 2 * c["bias_mc_se"] for c in cells)

    window = "two weeks"
    require("outcome window documented", "following two weeks" in e.design and "two-week window" in e.design)
    require("covariate window documented", "The covariates describe the twelve months before" in e.design)

    for number, title in enumerate(SECTIONS, start=1):
        add(f"## {number}. {title}\n", "document structure")
    add(f"shown in Section {SECTIONS.index('What randomization establishes') + 1}.", "document structure")

    # 1. Experiment
    add(f"over the {window} after the campaign. It is", "docs/design.md, Section 1.5")
    add(
        f"The Hillstrom (2008) experiment randomly assigned {data['rows']:,} customers",
        "provenance.json; configs/data.toml",
    )
    require("publication year", "(2008)" in config["data"]["dataset"]["name"])
    header = ["Contrast", "Role", "Outcome", "Estimand", "Estimate", f"{level} CI", "Relative lift", "Holm p"]
    add(row(*header, "Randomization p"), "configs/experiment.toml")
    for contrast, label in CONTRASTS.items():
        for outcome in OUTCOMES:
            r = dim.loc[(contrast, outcome)]
            add(
                row(
                    label,
                    r["role"],
                    outcome,
                    r["estimand"],
                    amount(r["estimate"], outcome),
                    interval(r["ci_low"], r["ci_high"], outcome),
                    percent(lift.loc[(contrast, outcome), "relative_lift"]),
                    p_value(r["p_holm"]),
                    p_value(r["p_randomization"], smallest),
                ),
                "experiment/effects.csv; experiment/relative_lift.csv",
            )
    co_primary = dim[dim["role"] == "co-primary"].index.get_level_values(0).nunique()
    add(
        f"Holm p-values adjust the {WORDS[co_primary]} co-primary contrasts for multiplicity; randomization "
        f"p-values come from {plan['permutations']:,} re-randomizations.",
        "experiment/effects.csv; configs/experiment.toml",
    )
    control = lift.loc[("mens_vs_control", "spend")]
    add(
        f"raised mean two-week spend by {effect('mens_vs_control')}, a {percent(control['relative_lift'])} "
        f"lift over the No E-Mail mean of {money(control['control_mean'])}.",
        "experiment/effects.csv; experiment/relative_lift.csv",
    )
    add(f"the **Womens e-mail** raised it by {effect('womens_vs_control')}.", EFFECTS)
    versus = dim.loc[("mens_vs_womens", "spend")]
    add(
        f"out-earned the Womens e-mail by {effect('mens_vs_womens')} (secondary contrast, randomization "
        f"p = {p_value(versus['p_randomization'], smallest)}).",
        EFFECTS,
    )
    add(
        f"Visit rates rose by {effect('mens_vs_control', 'visit')} (Mens) and "
        f"{effect('womens_vs_control', 'visit')} (Womens).",
        EFFECTS,
    )
    d = t["diagnostics"].set_index("contrast")
    auc = " and ".join(f"{d.loc[c, 'out_of_fold_auc']:.3f}" for c in PAIRS)
    likelihood = " and ".join(f"{d.loc[c, 'lr_p_value']:.2f}" for c in PAIRS)
    add(
        f"(out-of-fold AUC {auc}, likelihood-ratio p = {likelihood}, Mens and Womens), and the largest "
        f"absolute standardized mean difference is {t['balance']['smd'].abs().max():.3f}.",
        "experiment/diagnostics.csv; experiment/balance.csv",
    )
    boundary = next(c for c in e.validation["checks"] if c["name"] == "spend_upper_boundary")["details"]
    top = f"${boundary['maximum']:.0f}"
    r = t["robustness"]
    versions = r[(r["analysis"] == "spend version") & r["outcome"].str.startswith("spend")]
    recoded = versions["outcome"].str.extract(r"(\$\d+) recoded")[0].dropna().unique().tolist()
    require("recoded maximum", recoded == [top], str(recoded))
    span = versions.groupby("contrast")["estimate"].agg(["min", "max"])
    once = r[r["analysis"].str.startswith("without duplicate") & (r["outcome"] == "spend")].set_index(
        "contrast"
    )
    once = once["estimate"]
    stability = t["stability"]
    streams = sorted(stability["stream"].unique())
    require("bootstrap seeds", streams == config["robustness"]["randomness"]["bootstrap_stability_streams"])
    low = stability.loc[stability["contrast"] == "mens_vs_control", "ci_low"]
    mens, womens = (f"{money(span.loc[c, 'min'])} to {money(span.loc[c, 'max'])}" for c in PAIRS)
    add(
        f"Across the pre-specified spend versions (recoding the {top} values, winsorizing) the Mens estimate "
        f"ranges from {mens} and the Womens estimate from {womens}. Keeping one copy of each exact "
        f"duplicate row gives {money(once['mens_vs_control'])} and {money(once['womens_vs_control'])}. Under "
        f"{WORDS[len(streams)]} further bootstrap seeds the lower bound of the Mens BCa interval stays "
        f"between ${low.min():.3f} and ${low.max():.3f}.",
        f"robustness/experiment_robustness.csv; robustness/bootstrap_stability.csv; {VALIDATION}",
    )
    add(f"![ITT effects with {level} intervals]", "configs/experiment.toml")

    # 2. Constructed selection
    reference = benchmark["selection"]["reference"]
    require("sampling reference", reference.startswith("Keith") and "(2023)" in reference)
    add("(RCT rejection sampling, Keith et al. 2023)", FROZEN)
    add(
        f"{WORDS[len(benchmark['scenarios'])].capitalize()} scenarios, {draws} draws each:",
        "configs/benchmark/v1.toml; comparison/performance.csv",
    )
    strength = {name: s["selection_strength_k"] for name, s in benchmark["scenarios"].items()}
    require(
        "B is twice as strong as A", strength["B"] == 2 * strength["A"] and strength["C"] == strength["A"]
    )
    require("C hides recency", benchmark["scenarios"]["C"]["hidden_covariates"] == ["recency"])
    add("- **B**: twice as strong,", FROZEN)
    naive = t["decomposition"]
    naive = naive[(naive["scenario"] == "A") & (naive["outcome"] == "spend")].set_index("comparison")
    shares = " and ".join(
        f"{percent(naive.loc[c, 'bias_vs_ate'] / naive.loc[c, 'ate'])} ({n})" for c, n in PAIRS.items()
    )
    terms = "; ".join(
        f"{money(naive.loc[c, 'baseline_bias'])} and {money(naive.loc[c, 'heterogeneity_term'])} "
        f"per customer for {n}"
        for c, n in PAIRS.items()
    )
    add(f"the naive comparison overstates the spend effect by {shares}.", DECOMPOSITION)
    add(f"(ATT minus ATE, which can take either sign): {terms}.", DECOMPOSITION)

    # 3. Recovery
    recovery = {
        (s, o): [names[x] for x in ADJUSTED if unbiased(x, s, o)] for s in "ABC" for o in ("spend", "visit")
    }
    for s, label in (("A", "moderate selection"), ("B", "strong selection"), ("C", "recency hidden")):
        add(
            f"- **Scenario {s}** ({label}): spend: {listing(recovery[(s, 'spend')])}; "
            f"visit: {listing(recovery[(s, 'visit')])}.",
            PERFORMANCE,
        )
    cells = len(ADJUSTED) * len(PAIRS) * 2 * 2
    matching = [scored(c, s, "matching_nn", o) for c in PAIRS for s in "ABC" for o in ("spend", "visit")]
    require("matching is biased in every cell", all(abs(m["bias"]) > 2 * m["bias_mc_se"] for m in matching))
    require("nothing recovers in C", not recovery[("C", "spend")] and not recovery[("C", "visit")])
    tail = math.erfc(2 / math.sqrt(2))
    add(
        f"Each flag is a separate test at roughly the {percent(tail)} level, so among the {cells} feasible "
        f"cells in Scenarios A and B about {0.05 * cells:.0f} would be flagged by chance",
        "comparison/performance.csv; the two-SE rule",
    )
    rule = next(x for x in benchmark["estimators"] if x["name"] == "matching_nn")["matching"]
    add(
        f"1:{rule['ratio']} matching with a {rule['caliper_sd_of_logit']}-SD caliper is biased in every cell",
        "comparison/performance.csv; configs/benchmark/v1.toml",
    )
    add(
        f"come from one constructed dataset (draw {draw}, fixed by the frozen specification); bias and "
        f"coverage summarize all {draws} draws. `*` marks a bias larger than twice its Monte Carlo standard "
        "error.",
        "configs/benchmark/v1.toml; comparison/performance.csv",
    )
    add(
        row(
            "Estimator",
            "Estimand",
            f"Estimated effect (draw {draw})",
            f"{level} CI (draw {draw})",
            f"Mean bias vs benchmark ({draws} draws)",
            "CI coverage",
            "Key assumptions",
        ),
        FROZEN,
        times=2,
    )
    for comparison in PAIRS:
        for cells_ in estimator_rows(e, comparison, "A", "spend"):
            add(row(*cells_), "comparison/display_draw.csv; comparison/performance.csv")

    # 4. Heterogeneity and targeting
    splits = config["heterogeneity"]["blp"]["n_splits"]
    add(
        f"(the BLP slope b2, or the spread of sorted group effects, over {splits} sample splits)",
        "configs/heterogeneity.toml",
    )
    add(
        row("Contrast", "Outcome", "BLP slope b2", "p (split-adjusted)", "Heterogeneity detected"),
        VERDICTS,
    )
    for v in t["verdicts"].itertuples():
        detected = "yes" if v.heterogeneity_detected else "no"
        add(
            row(PAIRS[v.contrast], v.outcome, f"{v.b2:.2f}", p_value(v.b2_p_value), detected),
            VERDICTS,
        )
    found = [
        f"{PAIRS[v.contrast]} on {v.outcome}" for v in t["verdicts"].itertuples() if v.heterogeneity_detected
    ]
    add(f"Heterogeneity was detected only for {listing(found)}.", VERDICTS)
    policy = (
        t["policy"]
        .set_index(["policy", "versus", "outcome"])
        .loc[("model_best_arm", "treat_all_mens", "spend")]
    )
    require("targeting interval includes zero", policy["ci_low"] <= 0 <= policy["ci_high"])
    add(
        f"changes expected spend by {money(policy['difference'])} per customer, compared with sending "
        f"everyone the Mens e-mail ({level} CI {interval(policy['ci_low'], policy['ci_high'], 'spend')}).",
        "heterogeneity/policy_differences.csv",
    )

    # 5. Robustness and sensitivity
    ref = t["refutation"]
    placebo = ref.groupby("scenario")["placebo_rejection_rate"].mean()
    subset = re.search(r"random (\d+%) subsample", config["robustness"]["refutation"]["data_subset"]).group(1)
    first = config["robustness"]["refutation"]["n_draws"]
    require("refutation draws", set(ref["n_draws"]) == {first})
    refuted = ref[(ref["scenario"] == "C") & (ref["outcome"] == "spend")]
    adjusted_c = perf[
        (perf["scenario"] == "C")
        & (perf["outcome"] == "spend")
        & perf["estimator"].isin([x for x in ADJUSTED if x != "matching_nn"])
    ]
    adjusted_c = (adjusted_c["bias"] / adjusted_c["benchmark_value"]).abs()
    add(
        f"A placebo treatment is rejected in {percent(placebo['A'])} (A) and {percent(placebo['C'])} (C) of "
        f"draws on average, and adding a random covariate or refitting on {subset} subsets",
        "robustness/refutation_summary.csv; configs/robustness.toml",
    )
    methods = listing([names[x] for x in sorted(refuted["estimator"].unique())])
    subset_bias = refuted["original_bias_pct"].abs()
    add(
        f"Yet in the same first {first} draws of Scenario C, the {methods} spend estimates are biased by "
        f"{percent(subset_bias.min())} to {percent(subset_bias.max())}. Over all draws, every adjusted "
        f"estimator except matching is biased by {percent(adjusted_c.min())} to {percent(adjusted_c.max())} "
        f"on spend (Section {SECTIONS.index('Which estimators recover the experimental answer') + 1}).",
        f"robustness/refutation_summary.csv; {PERFORMANCE}",
    )
    s = t["sensitivity"].set_index(["comparison", "scenario"])
    values = {c: (s.loc[(c, "A"), "visit_rv_q1"], s.loc[(c, "C"), "visit_rv_q1"]) for c in PAIRS}
    require("robustness value is higher in C", all(c > a for a, c in values.values()))
    robust = " and ".join(f"{a:.3f} (A) vs {c:.3f} (C) for {PAIRS[k]}" for k, (a, c) in values.items())
    add(f"The robustness value for visit is {robust}:", SENSITIVITY)
    bounds = [k for k in (1, 2, 3) if f"visit_bound_k{k}_covers_error" in s.columns]
    covered = [percent(s.loc[("mens_vs_control", "C"), f"visit_bound_k{k}_covers_error"]) for k in bounds]
    times = f"{', '.join(WORDS[k] for k in bounds[:-1])} or {WORDS[bounds[-1]]}"
    add(f"Bounds for a confounder {times} times as strong", SENSITIVITY)
    add(
        f"relative to the experimental reference in {listing(covered)} of Mens draws",
        SENSITIVITY,
    )

    # 6. Coverage in C
    def coverage(estimator: str, outcome: str) -> str:
        return " and ".join(percent(scored(c, "C", estimator, outcome)["coverage"]) for c in PAIRS)

    add(
        f"covers the experimental reference in {coverage('aipw', 'visit')} of draws (Mens, Womens), against "
        f"{coverage('oracle_ipw', 'visit')} for the oracle,",
        PERFORMANCE,
    )
    add(
        f"the same intervals still cover in {coverage('aipw', 'spend')} of draws,",
        PERFORMANCE,
    )

    # 7 and 8. Assumptions and conclusions
    arms = dim.loc[("mens_vs_control", "spend")][["n_treated", "n_control"]].tolist()
    arms.append(dim.loc[("womens_vs_control", "spend"), "n_treated"])
    require("arms sum to the file", sum(arms) == data["rows"])
    require("about a third per arm", all(abs(a / data["rows"] - 1 / 3) < 0.01 for a in arms))
    add("each arm received about a third of customers.", EFFECTS)
    add(
        f"Covariates describe the prior twelve months; outcomes the {window} after the campaign.",
        "docs/design.md, Section 1.6",
    )
    require("both e-mails raised every outcome", (dim.loc[list(PAIRS), "ci_low"] > 0).all())
    require("Mens beat Womens on every outcome", (dim.loc["mens_vs_womens", "ci_low"] > 0).all())
    add(
        "increased two-week spend, visits and conversions, and the Mens e-mail outperformed the Womens "
        f"e-mail on all {WORDS[len(OUTCOMES)]}.",
        EFFECTS,
    )
    add(
        f"The data are from 2008 and the outcome window is {window}.",
        "configs/data.toml; docs/design.md, Section 1.5",
    )
    add(f"The data hold an observed maximum of {top};", "validation_raw.json")

    # 9 and 10. Figure guide and provenance
    add(f"is there structure at the {top} maximum?", "validation_raw.json")
    add(f"Across {draws} draws, how large is each estimator's bias", FROZEN)
    stamp = e.provenance["comparison"]
    packages = ", ".join(f"{k} {v}" for k, v in stamp["environment"]["packages"].items())
    b = stamp["benchmark"]
    add(
        f"- Processed data SHA-256 `{data['processed_sha256']}` ({data['rows']:,} rows, none removed); raw "
        f"data SHA-256 `{data['raw_sha256']}`.",
        "provenance.json",
    )
    add(
        f"- Benchmark specification v{b['version']} (SHA-256 `{b['sha256']}`), base seed {b['base_seed']}, "
        f"{b['n_draws']} draws.",
        "provenance.json; configs/benchmark/v1.toml",
    )
    add(f"- Code `{stamp['code']['commit'][:7]}`.", "provenance.json")
    add(f"- Python {stamp['environment']['python']}; {packages}.", "provenance.json")
    return claims


def readme_claims(e: Evidence) -> list[Claim]:
    t, perf = e.tables, e.tables["performance"]
    effects = t["effects"]
    dim = effects[(effects["estimator"] == "difference_in_means") & (effects["outcome"] == "spend")]
    dim = dim.set_index("contrast")["estimate"]
    rows = e.provenance["experiment"]["data"]["rows"]
    python = re.search(r">=(\d+\.\d+),", e.pyproject["project"]["requires-python"]).group(1)
    level = f"{e.configs['experiment']['inference']['ci_level']:.0%}"
    timing = re.search(r"the earlier timed run took (\d+) minutes on (\d+) cores", e.design)
    require("documented runtime", timing is not None)
    criteo = e.configs["criteo"]["source"]["name"]
    require("extension dataset", criteo.startswith("Criteo Uplift"))
    version = re.search(r"v\d+\.\d+", criteo).group(0)

    def biased(table: pd.DataFrame) -> pd.Series:
        return table["bias"].abs() > 2 * table["bias_mc_se"]

    weighting = perf["estimator"].isin(["ipw_ate", "ipw_att", "aipw"]) & perf["outcome"].isin(
        ["spend", "visit"]
    )
    require(
        "weighting and AIPW recover", not biased(perf[weighting & perf["scenario"].isin(["A", "B"])]).any()
    )
    hidden = perf[(perf["scenario"] == "C") & (perf["estimator"] != "oracle_ipw")]
    require("every method is biased in C", bool(biased(hidden).all()))
    return [
        Claim(f"The study uses the Hillstrom experiment, with {rows:,} customers.", "provenance.json"),
        Claim(f"Python {python}, pandas, scikit-learn, EconML, DoubleML.", "pyproject.toml"),
        Claim(
            f"increased two-week spending by about {money(dim['mens_vs_control'])} per customer; the Womens "
            f"email increased it by about {money(dim['womens_vs_control'])}.",
            EFFECTS,
        ),
        Claim(f"![Experimental effects with {level} intervals]", "configs/experiment.toml"),
        Claim(f"(Criteo Uplift {version})", "configs/extension/criteo_gate.toml"),
        Claim(f"with Python {python} installed:", "pyproject.toml"),
        Claim(
            f"The full analysis took about {timing.group(1)} minutes on {WORDS[int(timing.group(2))]} cores "
            "in the documented run.",
            "docs/design.md, Section 7 item 14 (a recorded statement, not a measured result)",
        ),
    ]


# Documents


def check_document(name: str, text: str, claims: list[Claim]) -> int:
    """Every claim must be present, and no number may be left without one. Returns the numbers checked."""
    masked = list(text)
    for claim in claims:
        starts = [m.start() for m in re.finditer(re.escape(claim.text), text)]
        require(
            f"{name} does not match {claim.source}",
            len(starts) == claim.times,
            f"expected {claim.times}x, found {len(starts)}x: {claim.text}",
        )
        for start in starts:
            masked[start : start + len(claim.text)] = "\0" * len(claim.text)
    rest = "".join(masked)
    for number, line in enumerate(rest.splitlines(), start=1):
        found = NUMBER.search(line)
        require(
            f"{name} line {number} quotes a number with no source",
            found is None,
            text.splitlines()[number - 1][:160],
        )
    return len(re.findall(r"\d[\d,.]*", text))


def run(root: Path) -> dict[str, int]:
    evidence = load(root)
    check_provenance(root, evidence)
    check_tables(evidence)
    check_estimator_table(evidence)
    checked = {}
    for name, build in (("README.md", readme_claims), ("docs/findings.md", findings_claims)):
        checked[name] = check_document(name, (root / name).read_text(), build(evidence))
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--list", action="store_true", help="print the source of every checked statement")
    args = parser.parse_args()
    try:
        checked = run(args.root)
    except AuditError as error:
        print(f"Audit failed: {error}")
        return 1
    if args.list:
        evidence = load(args.root)
        for name, build in (("README.md", readme_claims), ("docs/findings.md", findings_claims)):
            for claim in build(evidence):
                print(f"{name}\t{claim.source}\t{claim.text}")
    totals = "; ".join(f"{name}: {count} numbers" for name, count in checked.items())
    print(f"Audit passed against {len(COMMITTED)} committed files. {totals}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
