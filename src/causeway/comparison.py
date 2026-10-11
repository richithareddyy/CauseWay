"""Milestone 7: compare every observational estimator against its estimand-matched experimental benchmark.

Two views are kept separate:
- one realized observational dataset (the pre-registered display draw), which is what an analyst would
  actually face: an estimate, an interval, and an error that only the benchmark reveals;
- the full Monte Carlo distribution over 200 draws, which characterizes each estimator's bias, variance
  and interval calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causeway.benchmark import build_pair, experimental_benchmarks, monte_carlo_summary
from causeway.config import DataPaths, load_benchmark_config
from causeway.pipeline import load_processed
from causeway.provenance import read_provenance

REFERENCE_ESTIMATORS = ["naive_difference", "oracle_ipw"]
ESTIMATOR_ORDER = [
    "naive_difference",
    "oracle_ipw",
    "outcome_regression",
    "ipw_ate",
    "aipw",
    "dml_irm",
    "ipw_att",
    "matching_nn",
]
ROLE = {
    "naive_difference": "reference: no adjustment",
    "oracle_ipw": "reference: true propensity, not available to an analyst",
}
ASSUMPTIONS = {
    "experiment": "Randomized assignment; SUTVA (no interference, one version of each e-mail).",
    "naive_difference": "Assignment independent of potential outcomes (violated by construction in Part II).",
    "oracle_ipw": "Known propensity; positivity. Checks the construction; not a feasible estimator.",
    "outcome_regression": "Unconfoundedness given the analyst covariates; correctly specified outcome model.",
    "ipw_ate": "Unconfoundedness; positivity for both arms; correctly specified propensity model.",
    "aipw": "Unconfoundedness; positivity; at least one of the propensity and outcome models correct.",
    "dml_irm": "Unconfoundedness; positivity; nuisance learners converge fast enough (o(n^-1/4)) under "
    "cross-fitting.",
    "ipw_att": "Unconfoundedness for controls; controls overlap the treated; correct propensity model.",
    "matching_nn": "Unconfoundedness; treated units have close controls; unmatched treated change the "
    "estimand to the ATT among matched treated.",
}
PROVENANCE_KEYS = [
    ("data", "processed_sha256"),
    ("benchmark", "sha256"),
    ("benchmark", "base_seed"),
    ("benchmark", "n_draws"),
]


class ProvenanceMismatch(RuntimeError):
    """Outputs being combined were produced from different data, specifications or seeds."""


def _results_dir(paths: DataPaths, name: str):
    return paths.root / "reports" / "results" / name


def check_provenance(records: dict[str, dict]) -> dict:
    """Every source must share the data checksum, benchmark specification, seed and number of draws."""
    reference_name, reference = next(iter(records.items()))
    for name, record in records.items():
        for section, key in PROVENANCE_KEYS:
            if record.get(section, {}).get(key) != reference.get(section, {}).get(key):
                raise ProvenanceMismatch(f"{name} and {reference_name} differ in {section}.{key}")
    return reference


def load_benchmark_estimates(paths: DataPaths, n_draws: int) -> tuple[pd.DataFrame, dict]:
    """Combine Milestone 5 and 6 estimates after checking provenance, shared references and completeness."""
    sources = ["propensity", "estimation"]
    records = {name: read_provenance(_results_dir(paths, name)) for name in sources}
    provenance = check_provenance(records)
    frames = {name: pd.read_csv(_results_dir(paths, name) / "estimates.csv") for name in sources}

    keys = ["comparison", "scenario", "draw", "outcome", "estimator"]
    shared = [
        frame[frame["estimator"].isin(REFERENCE_ESTIMATORS)].set_index(keys)["estimate"].sort_index()
        for frame in frames.values()
    ]
    if not shared[0].index.equals(shared[1].index) or not np.allclose(
        shared[0], shared[1], rtol=0, atol=1e-12
    ):
        raise ProvenanceMismatch("the reference estimators differ between the propensity and estimation runs")

    combined = pd.concat(
        [
            frames["propensity"],
            frames["estimation"][~frames["estimation"]["estimator"].isin(REFERENCE_ESTIMATORS)],
        ],
        ignore_index=True,
    )
    counts = combined.groupby(["comparison", "scenario", "outcome", "estimator"]).size()
    incomplete = counts[counts != n_draws]
    if len(incomplete) or set(combined["estimator"]) != set(ESTIMATOR_ORDER):
        raise ProvenanceMismatch(
            f"incomplete estimates: {incomplete.to_dict()}; estimators {sorted(set(combined['estimator']))}"
        )
    return combined, provenance


def performance_table(estimates: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    """Monte Carlo metrics for every estimator, with the oracle's coverage alongside as the reference."""
    summary = monte_carlo_summary(estimates, benchmarks)
    oracle = summary[summary["estimator"] == "oracle_ipw"].set_index(["comparison", "scenario", "outcome"])
    summary["oracle_coverage"] = [
        oracle.loc[(c, s, o), "coverage"]
        for c, s, o in zip(summary["comparison"], summary["scenario"], summary["outcome"], strict=True)
    ]
    summary["coverage_minus_oracle"] = summary["coverage"] - summary["oracle_coverage"]
    summary["bias_pct"] = summary["bias"] / summary["benchmark_value"]
    summary["unbiased_within_mc_error"] = summary["bias"].abs() <= 2 * summary["bias_mc_se"]
    summary["role"] = summary["estimator"].map(ROLE).fillna("candidate estimator")
    summary["key_assumptions"] = summary["estimator"].map(ASSUMPTIONS)
    order = {name: i for i, name in enumerate(ESTIMATOR_ORDER)}
    return summary.sort_values(
        ["comparison", "scenario", "outcome", "estimator"],
        key=lambda column: column.map(order) if column.name == "estimator" else column,
        kind="stable",
    ).reset_index(drop=True)


def display_table(
    estimates: pd.DataFrame, benchmarks: pd.DataFrame, experiment_effects: pd.DataFrame, draw: int
) -> pd.DataFrame:
    """One realized observational dataset: each estimate beside its estimand-matched benchmark.

    The first rows per cell are the randomized-experiment estimates (Part I) for the full pair, with their
    own 95% intervals, followed by the experimental ATT benchmark for that scenario.
    """
    rows = []
    shown = estimates[estimates["draw"] == draw]
    for (comparison, scenario, outcome), group in shown.groupby(
        ["comparison", "scenario", "outcome"], sort=False
    ):
        targets = benchmarks[
            (benchmarks["comparison"] == comparison)
            & (benchmarks["scenario"] == scenario)
            & (benchmarks["outcome"] == outcome)
        ].set_index("estimand")
        experiment = experiment_effects[
            (experiment_effects["contrast"] == comparison)
            & (experiment_effects["outcome"] == outcome)
            & (experiment_effects["estimator"] == "difference_in_means")
        ].iloc[0]
        base = {"comparison": comparison, "scenario": scenario, "outcome": outcome}
        rows.append(
            {
                **base,
                "estimator": "experiment",
                "estimand": "ATE (randomized, full pair)",
                "estimate": experiment["estimate"],
                "ci_low": experiment["ci_low"],
                "ci_high": experiment["ci_high"],
                "benchmark_value": targets.loc["ATE", "value"],
                "role": "experimental benchmark",
                "key_assumptions": ASSUMPTIONS["experiment"],
            }
        )
        att = targets.loc["ATT"]
        rows.append(
            {
                **base,
                "estimator": "experiment_att",
                "estimand": "ATT (randomized pair, e*-weighted)",
                "estimate": att["value"],
                "ci_low": att["value"] - 1.959964 * att["se"],
                "ci_high": att["value"] + 1.959964 * att["se"],
                "benchmark_value": att["value"],
                "role": "experimental benchmark",
                "key_assumptions": ASSUMPTIONS["experiment"],
            }
        )
        for estimator in ESTIMATOR_ORDER:
            row = group[group["estimator"] == estimator].iloc[0]
            benchmark = targets.loc[row["benchmark"], "value"]
            rows.append(
                {
                    **base,
                    "estimator": estimator,
                    "estimand": row["estimand"],
                    "estimate": row["estimate"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "benchmark_value": benchmark,
                    "role": ROLE.get(estimator, "candidate estimator"),
                    "key_assumptions": ASSUMPTIONS[estimator],
                }
            )
    table = pd.DataFrame(rows)
    table["error_vs_benchmark"] = table["estimate"] - table["benchmark_value"]
    table["error_pct"] = table["error_vs_benchmark"] / table["benchmark_value"]
    table["ci_covers_benchmark"] = (table["ci_low"] <= table["benchmark_value"]) & (
        table["benchmark_value"] <= table["ci_high"]
    )
    return table


def load_comparison_inputs(paths: DataPaths | None = None) -> dict:
    paths = paths or DataPaths()
    spec = load_benchmark_config(1)
    estimates, provenance = load_benchmark_estimates(paths, spec["randomness"]["n_draws"])
    experiment_provenance = read_provenance(_results_dir(paths, "experiment"))
    if experiment_provenance["data"]["processed_sha256"] != provenance["data"]["processed_sha256"]:
        raise ProvenanceMismatch("the Part I results were produced from different processed data")
    frame = load_processed(paths)
    benchmarks = pd.concat(
        [experimental_benchmarks(build_pair(frame, c, spec), spec) for c in spec["comparisons"]],
        ignore_index=True,
    )
    experiment_effects = pd.read_csv(_results_dir(paths, "experiment") / "effects.csv")
    return {
        "spec": spec,
        "estimates": estimates,
        "benchmarks": benchmarks,
        "provenance": provenance,
        "experiment_effects": experiment_effects,
    }
