"""End-to-end Part I report on the synthetic fixture: every output is produced from computed values."""

import pandas as pd
import pytest

from causeway.reporting import run_experiment_report


@pytest.fixture
def outputs(data_config, paths):
    from causeway.pipeline import run_data_pipeline

    run_data_pipeline(data_config, paths)
    return run_experiment_report(paths)


def test_report_files_and_figures_exist(outputs, paths):
    results_dir = paths.root / "reports" / "results" / "experiment"
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
