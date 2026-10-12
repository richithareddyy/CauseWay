"""README and docs/: links resolve, and the published figures are the ones the code publishes."""

import re
from pathlib import Path

from causeway import findings
from causeway.config import PROJECT_ROOT

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def _relative_links(markdown: Path) -> list[str]:
    targets = [t.split("#")[0] for t in LINK.findall(markdown.read_text())]
    return [t for t in targets if t and not t.startswith(("http://", "https://", "mailto:"))]


def test_readme_and_docs_links_resolve():
    for markdown in [PROJECT_ROOT / "README.md", *(PROJECT_ROOT / "docs").glob("*.md")]:
        for target in _relative_links(markdown):
            assert (markdown.parent / target).exists(), f"{markdown.name} links to missing {target}"


def test_published_figures_match_the_code():
    published = {Path(p).name for p in findings.PUBLISHED_FIGURES}
    assert len(published) == len(findings.PUBLISHED_FIGURES), "published figure names must be unique"
    assert {p.name for p in (PROJECT_ROOT / "docs" / "figures").glob("*.png")} == published
    readme_images = re.findall(
        r"!\[[^\]]*\]\(docs/figures/([^)]+)\)", (PROJECT_ROOT / "README.md").read_text()
    )
    assert readme_images and set(readme_images) <= published


def test_relink_points_published_figures_at_docs_and_keeps_others_as_paths():
    markdown = (
        "![ITT](../../figures/experiment/itt_effects.png)\n"
        "| [benchmark/selection_imbalance.png](../../figures/benchmark/selection_imbalance.png) |\n"
        "| [experiment/outcome_levels_by_arm.png](../../figures/experiment/outcome_levels_by_arm.png) |"
    )
    assert findings.relink(markdown).splitlines() == [
        "![ITT](figures/itt_effects.png)",
        "| [benchmark/selection_imbalance.png](figures/selection_imbalance.png) |",
        "| `reports/figures/experiment/outcome_levels_by_arm.png` |",
    ]


def test_publish_docs_copies_the_report_and_figures(paths, tmp_path):
    report = paths.results("findings") / "findings_report.md"
    report.parent.mkdir(parents=True)
    report.write_text("# Causeway: findings\n\nBody ![x](../../figures/experiment/itt_effects.png)\n")
    for path in findings.PUBLISHED_FIGURES:
        paths.figures(path).parent.mkdir(parents=True, exist_ok=True)
        paths.figures(path).write_bytes(b"png")
    docs = tmp_path / "docs"
    written = findings.publish_docs(paths, docs)
    assert len(written) == 1 + len(findings.PUBLISHED_FIGURES)
    text = (docs / "findings.md").read_text()
    assert text.startswith("# Causeway: findings\n\n> Copied from")
    assert "![x](figures/itt_effects.png)" in text
