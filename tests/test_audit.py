"""The committed-results audit: it passes on the committed files and rejects anything altered."""

import json
import re
import shutil
import subprocess

import pandas as pd
import pytest

from causeway import audit
from causeway.config import PROJECT_ROOT

DOCUMENTS = {"README.md": audit.readme_claims, "docs/findings.md": audit.findings_claims}


@pytest.fixture
def saved_study(tmp_path):
    """Only what a fresh clone has: no data/ directory and no uncommitted results."""
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    for name in [*audit.COMMITTED, "README.md", "docs/findings.md", "docs/design.md", "pyproject.toml"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / name, tmp_path / name)
    return tmp_path


def test_audit_passes_on_committed_files_without_raw_data(saved_study):
    checked = audit.run(saved_study)
    assert set(checked) == set(DOCUMENTS) and all(checked.values())
    assert not (saved_study / "data").exists()


@pytest.mark.parametrize("name", DOCUMENTS)
def test_changing_any_single_number_is_caught(name):
    text = (PROJECT_ROOT / name).read_text()
    claims = DOCUMENTS[name](audit.load(PROJECT_ROOT))
    audit.check_document(name, text, claims)
    missed = []
    for match in re.finditer(r"\d+", text):
        last = match.group()[-1]
        altered = match.group()[:-1] + str((int(last) + 1) % 10)
        try:
            audit.check_document(name, text[: match.start()] + altered + text[match.end() :], claims)
        except audit.AuditError:
            continue
        missed.append((match.group(), text[max(0, match.start() - 40) : match.end() + 20]))
    assert not missed


@pytest.mark.parametrize("name", DOCUMENTS)
@pytest.mark.parametrize("sentence", ["The effect was 3.2 times larger.", "It was two times larger."])
def test_a_number_without_a_source_is_caught(name, sentence):
    text = (PROJECT_ROOT / name).read_text()
    with pytest.raises(audit.AuditError, match="no source"):
        audit.check_document(name, f"{text}\n{sentence}\n", DOCUMENTS[name](audit.load(PROJECT_ROOT)))


@pytest.mark.parametrize("table", [name for name in audit.TABLES if name != "estimator_table"])
def test_every_committed_table_is_actually_used(saved_study, table):
    """Shifting a table's numbers must break the audit, otherwise the table supports nothing."""
    path = saved_study / audit.TABLES[table]
    frame = pd.read_csv(path)
    numeric = [c for c in frame.select_dtypes("number").columns if c not in ("n_draws", "stream")]
    frame[numeric] = frame[numeric].astype(float) * 1.5 + 0.1
    frame.to_csv(path, index=False)
    with pytest.raises(audit.AuditError):
        audit.run(saved_study)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("estimator_table", "estimator table"),
        ("biased_flag", "bias flag"),
        ("unquoted_column", "mean estimate"),
        ("data_checksum", "provenance data"),
        ("dirty_tree", "provenance"),
        ("benchmark_config", "benchmark configuration"),
        ("row_level_table", "not an aggregate"),
        ("missing_file", "missing"),
        ("stale_findings", "docs/findings.md"),
    ],
)
def test_corrupt_evidence_is_rejected(saved_study, mutation, expected):
    performance = saved_study / audit.TABLES["performance"]
    provenance = saved_study / "reports/results/robustness/provenance.json"
    if mutation == "estimator_table":
        path = saved_study / audit.TABLES["estimator_table"]
        path.write_text(path.read_text().replace("+48.1% *", "+38.1% *", 1))
    elif mutation == "biased_flag":
        frame = pd.read_csv(performance)
        frame.loc[0, "unbiased_within_mc_error"] = not frame.loc[0, "unbiased_within_mc_error"]
        frame.to_csv(performance, index=False)
    elif mutation == "unquoted_column":
        frame = pd.read_csv(performance)
        frame.loc[5, "mean_estimate"] *= 1.01
        frame.to_csv(performance, index=False)
    elif mutation == "data_checksum":
        record = json.loads(provenance.read_text())
        record["data"]["raw_sha256"] = "0" * 64
        provenance.write_text(json.dumps(record))
    elif mutation == "dirty_tree":
        for source in audit.SOURCES:
            path = saved_study / f"reports/results/{source}/provenance.json"
            record = json.loads(path.read_text())
            record["code"]["uncommitted_changes"] = True
            path.write_text(json.dumps(record))
    elif mutation == "benchmark_config":
        path = saved_study / "configs/benchmark/v1.toml"
        path.write_text(path.read_text() + "\n# edited\n")
    elif mutation == "row_level_table":
        frame = pd.read_csv(performance)
        pd.concat([frame] * 10).to_csv(performance, index=False)
    elif mutation == "missing_file":
        performance.unlink()
    else:
        path = saved_study / "docs/findings.md"
        path.write_text(path.read_text().replace("$0.77", "$0.78", 1))
    with pytest.raises(audit.AuditError, match=expected):
        audit.run(saved_study)


def test_findings_text_is_what_the_committed_tables_render():
    """The report generator, fed only committed files, reproduces the published text."""
    from causeway import findings
    from causeway.experiment import load_experiment_plan

    evidence = audit.load(PROJECT_ROOT)
    t = evidence.tables
    results = {
        "provenance": evidence.provenance,
        "summary": {"plan": load_experiment_plan(), "diagnostics": t["diagnostics"].to_dict("records")},
        "effects": t["effects"],
        "lift": t["lift"],
        "balance": t["balance"],
        "decomposition": t["decomposition"],
        "display": t["display"],
        "performance": t["performance"],
        "verdicts": t["verdicts"],
        "policy_differences": t["policy"],
        "experiment_robustness": t["robustness"],
        "bootstrap_stability": t["stability"],
        "refutation": t["refutation"],
        "sensitivity": t["sensitivity"],
    }
    guide = [(i, f"[{p}](../../figures/{p})", f.__doc__.strip()) for i, p, f in findings.FIGURE_GUIDE]
    tables = {
        "experiment_table": findings.experiment_table(t["effects"], t["lift"]),
        "estimator_table": findings.estimator_table(t["display"], t["performance"]),
        "assumptions": pd.DataFrame(
            [(name, *values) for name, values in findings.ASSUMPTIONS.items()],
            columns=["Assumption", "Status", "Part I", "Part II", "How it is examined"],
        ),
        "figure_guide": pd.DataFrame(
            guide, columns=["Required visualization", "Figure", "Question it answers"]
        ),
    }
    _, _, body = findings.findings_markdown(results, tables).partition("\n")
    published = (PROJECT_ROOT / "docs" / "findings.md").read_text()
    assert published.endswith(findings.relink(body.lstrip()))


def test_only_the_named_aggregates_are_tracked():
    listing = subprocess.run(
        ["git", "ls-files", "reports", "data"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    if listing.returncode:
        pytest.skip("not a git checkout")
    tracked = {name for name in listing.stdout.split() if not name.endswith(".gitkeep")}
    assert tracked == set(audit.COMMITTED)
