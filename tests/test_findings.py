"""The final findings report and the figures added for it."""

import itertools
import json

import pandas as pd
import pytest

from causeway import findings
from causeway import visualization as viz
from causeway.comparison import ProvenanceMismatch


def test_effects_are_written_in_their_own_units():
    assert findings._effect(0.7698, "spend") == "$0.77"
    assert findings._effect(-0.0776, "spend") == "-$0.08"
    assert findings._effect(0.0766, "visit") == "7.66 pp"
    assert findings._interval(-0.25, 0.1, "spend") == "[-$0.25, $0.10]"
    assert findings._interval(0.07, 0.0832, "visit") == "[7.00, 8.32]"


def _performance() -> pd.DataFrame:
    rows = []
    for comparison, scenario, outcome, estimator in itertools.product(
        ["mens_vs_control", "womens_vs_control"], "ABC", ["spend", "visit"], findings.RECOVERY_ESTIMATORS
    ):
        biased = estimator == "naive_difference" or (scenario == "C" and estimator != "oracle_ipw")
        rows.append(
            {
                "comparison": comparison,
                "scenario": scenario,
                "outcome": outcome,
                "estimator": estimator,
                "estimand": "ATT" if estimator in ("ipw_att", "matching_nn") else "ATE",
                "bias_pct": 0.3 if biased else 0.01,
                "coverage": 0.5 if biased else 0.95,
                "unbiased_within_mc_error": not biased,
            }
        )
    return pd.DataFrame(rows)


def test_estimators_are_listed_as_recovering_only_when_unbiased_in_both_comparisons():
    perf = _performance()
    assert "AIPW (linear, cross-fitted)" in findings._recovering(perf, "spend", "A")
    assert findings._recovering(perf, "spend", "C") == []
    # the oracle and the naive difference are references, never candidates
    assert not {"Oracle IPW (true e*)", "Naive difference"} & set(findings._recovering(perf, "visit", "A"))

    perf.loc[
        (perf["comparison"] == "womens_vs_control") & (perf["estimator"] == "aipw"),
        "unbiased_within_mc_error",
    ] = False
    assert "AIPW (linear, cross-fitted)" not in findings._recovering(perf, "spend", "A")


def test_estimator_table_keeps_each_estimand_and_flags_bias():
    perf = _performance()
    display = perf[["comparison", "scenario", "outcome", "estimator", "estimand"]].assign(
        estimate=1.0, ci_low=0.5, ci_high=1.5, key_assumptions="assumptions"
    )
    benchmark = display.drop_duplicates(["comparison", "scenario", "outcome"]).assign(
        estimator="experiment", estimand="ATE (randomized, full pair)"
    )
    table = findings.estimator_table(pd.concat([benchmark, display]), perf)
    assert {"Estimator", "Estimand", "95% CI (draw 0)", "Key assumptions"} <= set(table.columns)
    assert (
        table.loc[table["Estimator"] == "Randomized experiment", "Mean bias vs benchmark (200 draws)"]
        == "benchmark"
    ).all()
    naive = table[table["Estimator"] == "Naive difference"]
    assert naive["Mean bias vs benchmark (200 draws)"].str.endswith("*").all()
    matching = table[table["Estimator"] == "1:1 NN matching (ATT)"]
    assert (matching["Estimand"] == "ATT").all()
    # the experiment comes first in each block
    first = table.groupby(["comparison", "scenario", "outcome"], sort=False)["Estimator"].first()
    assert (first == "Randomized experiment").all()


def test_results_from_different_data_are_not_combined(paths):
    for i, name in enumerate(findings.SOURCES):
        directory = paths.results(name)
        directory.mkdir(parents=True)
        record = {"data": {"processed_sha256": "a" if i else "b"}}
        (directory / "provenance.json").write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="different processed datasets"):
        findings.load_results(paths)


def test_results_from_different_benchmark_specifications_are_not_combined(paths):
    for i, name in enumerate(findings.SOURCES):
        directory = paths.results(name)
        directory.mkdir(parents=True)
        record = {"data": {"processed_sha256": "a"}}
        if name in ("benchmark", "comparison"):
            record["benchmark"] = {"sha256": str(i), "base_seed": 1, "n_draws": 200}
        (directory / "provenance.json").write_text(json.dumps(record))
    with pytest.raises(ProvenanceMismatch):
        findings.load_results(paths)


def test_every_figure_in_the_guide_states_its_question():
    for _, path, function in findings.FIGURE_GUIDE:
        assert function.__doc__ and "?" in function.__doc__, path
        assert path.endswith(".png")


def test_recovery_heatmap_renders(tmp_path):
    path = viz.recovery_heatmap(_performance(), tmp_path / "recovery.png", findings.RECOVERY_ESTIMATORS)
    assert path.exists() and path.stat().st_size > 0


def test_segment_labels_are_readable_and_bands_sort_numerically():
    assert viz._segment_label("history_segment", "1) $0 - $100") == "Past-year spend: $0 - $100"
    assert viz._segment_label("newbie", 1) == "New customer: yes"
    assert viz._segment_label("zip_code", "Rural") == "ZIP area: Rural"
    levels = pd.DataFrame(
        {
            "variable": ["channel", "recency_band", "recency_band", "recency_band"],
            "level": ["Web", "1-3", "10-12", "4-6"],
        }
    )
    assert [lv for _, lv in viz._segment_rows(levels)] == ["Web", "1-3", "4-6", "10-12"]
