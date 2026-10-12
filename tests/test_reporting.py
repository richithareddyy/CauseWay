"""End-to-end Part I report on the synthetic fixture: every output is produced from computed values."""

import numpy as np
import pandas as pd
import pytest

from causeway.reporting import run_experiment_report


@pytest.fixture
def outputs(data_config, paths):
    from causeway.pipeline import run_data_pipeline

    run_data_pipeline(data_config, paths)
    return run_experiment_report(paths)


def test_report_files_and_figures_exist(outputs, paths):
    results_dir = paths.results("experiment")
    for name in ["effects.csv", "binary.csv", "balance.csv", "summary.json", "experiment_report.md"]:
        assert (results_dir / name).exists()
    assert len(outputs.figures) == 5 and all(f.exists() and f.stat().st_size > 0 for f in outputs.figures)


def test_report_numbers_come_from_the_effects_table(outputs):
    report = outputs.report_path.read_text()
    effects = pd.read_csv(outputs.report_path.parent / "effects.csv")
    row = effects[
        (effects["contrast"] == "mens_vs_control")
        & (effects["outcome"] == "spend")
        & (effects["estimator"] == "difference_in_means")
    ].iloc[0]
    assert f"| Mens vs No E-Mail | Difference in means | {row['estimate']:.3f} |" in report
    assert "intention-to-treat effect of assignment" in report
    assert "receiving" not in report.split("## Interpretation")[0]


def test_propensity_report_escapes_pipes_in_table_cells():
    from causeway.config import load_benchmark_config
    from causeway.reporting import _propensity_markdown

    diagnostics = pd.DataFrame(
        [
            {
                "comparison": "mens_vs_control",
                "scenario": "A",
                "diagnostic": "Max |SMD| x",
                "mean": 0.1,
                "p2_5": 0.0,
                "p97_5": 0.2,
            }
        ]
    )
    summary = pd.DataFrame(columns=["comparison"])
    report = _propensity_markdown(load_benchmark_config(1), summary, diagnostics, {"python": "3.12"})
    assert "Max \\|SMD\\| x" in report and "\\\\|" not in report


def test_heterogeneity_report_renders_every_figure_from_saved_tables(paths):
    """Rendering uses only saved tables, so a figure bug never forces the 16-minute computation to rerun."""
    from causeway.reporting import HETEROGENEITY_TABLES, render_heterogeneity_report

    rng = np.random.default_rng(0)
    results = paths.results("heterogeneity")
    results.mkdir(parents=True)
    paths.processed.mkdir(parents=True)
    (paths.processed / "manifest.json").write_text(
        '{"raw": {"sha256": "r"}, "processed": {"sha256": "p", "rows": 1}}'
    )
    contrasts, outcomes = ["mens_vs_control", "womens_vs_control"], ["spend", "visit", "conversion"]
    params = ["b1", "b2", *[f"gate_{g}" for g in range(1, 6)], "gate_top_minus_bottom"]
    gates = pd.DataFrame(
        [
            {
                "contrast": c,
                "outcome": o,
                "parameter": p,
                "estimate": 0.5,
                "ci_low": 0.1,
                "ci_high": 0.9,
                "p_value": 0.5 if p != "b2" else 0.0001,
                "n_splits": 100,
            }
            for c in contrasts
            for o in outcomes
            for p in params
        ]
    )
    verdicts = pd.DataFrame(
        [
            {
                "contrast": c,
                "outcome": o,
                "b2": 0.5,
                "b2_p_value": 0.0001,
                "gates_spread": 0.2,
                "gates_spread_ci_low": 0.1,
                "gates_spread_ci_high": 0.3,
                "blp_evidence": True,
                "gates_evidence": True,
                "heterogeneity_detected": True,
            }
            for c in contrasts
            for o in outcomes
        ]
    )
    levels = pd.DataFrame(
        [
            {
                "contrast": c,
                "outcome": o,
                "variable": v,
                "level": lv,
                "n_treated": 10,
                "n_control": 10,
                "estimate": 0.4,
                "se": 0.1,
                "ci_low": 0.2,
                "ci_high": 0.6,
                "p_value": 0.01,
                "mde": 0.3,
                "q_value": q,
                "classification": k,
            }
            for c in contrasts
            for o in outcomes
            for v, lv, q, k in (
                ("channel", "Web", 0.01, "benefit"),
                ("channel", "Phone", 0.5, "inconclusive"),
                ("newbie", "1", 0.01, "harm"),
            )
        ]
    )
    tests = pd.DataFrame(
        [
            {
                "contrast": c,
                "outcome": o,
                "variable": v,
                "chi2": 1.0,
                "dof": 1,
                "p_value": 0.3,
                "q_value": 0.4,
            }
            for c in contrasts
            for o in outcomes
            for v in ("channel", "newbie")
        ]
    )
    forest = pd.DataFrame(
        [
            {
                "contrast": c,
                "outcome": o,
                "mean": 0.5,
                "p5": 0.1,
                "p95": 0.9,
                "share_significant": 0.2,
                "top_features": "a, b, c",
            }
            for c in contrasts
            for o in ("spend", "visit")
        ]
    )
    cate = pd.concat(
        [
            pd.DataFrame(
                {"contrast": c, "outcome": o, "cate": rng.normal(size=50), "ci_low": -1.0, "ci_high": 1.0}
            )
            for c in contrasts
            for o in ("spend", "visit")
        ]
    )
    curves = pd.DataFrame(
        [
            {
                "email": e,
                "fraction_targeted": q,
                "model_incremental_value": q,
                "random_incremental_value": q,
                "outcome": o,
            }
            for e in ("mens", "womens")
            for q in (0.0, 0.5, 1.0)
            for o in ("spend", "visit")
        ]
    )
    values = pd.DataFrame(
        [
            {
                "policy": "treat_all_mens",
                "share_emailed": 1.0,
                "incremental_value": 0.7,
                "ci_low": 0.5,
                "ci_high": 0.9,
                "outcome": "spend",
            }
        ]
    )
    differences = pd.DataFrame(
        [
            {
                "policy": "model_best_arm",
                "versus": "treat_all_mens",
                "difference": -0.1,
                "ci_low": -0.3,
                "ci_high": 0.1,
                "interval_excludes_zero": False,
                "outcome": "spend",
            }
        ]
    )
    tables = {
        "blp_gates_clan": gates,
        "verdicts": verdicts,
        "segment_levels": levels,
        "segment_tests": tests,
        "forest_summary": forest,
        "uplift_curves": curves,
        "policy_values": values,
        "policy_differences": differences,
        "forest_cate": cate,
    }
    assert set(tables) == set(HETEROGENEITY_TABLES)
    for name, table in tables.items():
        table.to_csv(results / f"{name}.csv", index=False)
    outputs = render_heterogeneity_report(paths)
    assert len(outputs.figures) == 6 and all(f.exists() and f.stat().st_size > 0 for f in outputs.figures)
    assert "Heterogeneity detected" in outputs.report_path.read_text()


def test_robustness_report_runs_end_to_end(paths, monkeypatch):
    from causeway import reporting, robustness
    from causeway.processing import TransformationLog, build_interim, build_processed
    from conftest import make_raw_frame

    log = TransformationLog()
    frame = build_processed(build_interim(make_raw_frame(n_per_arm=1500, seed=13), log), log)
    monkeypatch.setattr(reporting, "load_processed", lambda _: frame)
    plan = robustness.load_robustness_plan()
    plan["benchmark"]["n_draws"] = plan["refutation"]["n_draws"] = 1
    plan["randomness"]["bootstrap_stability_streams"] = [3001]
    monkeypatch.setattr(robustness, "load_robustness_plan", lambda: plan)
    paths.processed.mkdir(parents=True)
    (paths.processed / "manifest.json").write_text(
        '{"raw": {"sha256": "r"}, "processed": {"sha256": "p", "rows": 1}}'
    )

    outputs = reporting.run_robustness_report(paths, n_jobs=1)
    assert outputs.figures and all(f.exists() and f.stat().st_size > 0 for f in outputs.figures)
    summary = pd.read_csv(paths.results("robustness") / "specification_summary.csv")
    boosted = summary[summary["estimator"] == "outcome_regression"]
    assert boosted["bias"].notna().all() and (boosted["failure_rate"] == 0).all()
