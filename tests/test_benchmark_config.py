"""Guards on the frozen observational benchmark specification (configs/benchmark/v1.toml)."""

import hashlib
import tomllib
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark" / "v1.toml"
# Recorded when v1 was frozen. A legitimate change must create v2.toml, never edit v1.
FROZEN_V1_SHA256 = "d88ce1a03de54eed5c8a38e095a49d01c0e07f0d12f95e66d3554f23b6f6697e"


@pytest.fixture(scope="module")
def spec() -> dict:
    return tomllib.loads(CONFIG.read_text())


def test_v1_file_is_unchanged_since_freeze():
    canonical = CONFIG.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical).hexdigest() == FROZEN_V1_SHA256


def test_preregistered_randomness(spec):
    assert spec["meta"]["status"] == "frozen"
    assert spec["randomness"]["base_seed"] == 20080320
    assert spec["randomness"]["n_draws"] == 200
    assert spec["randomness"]["fixed_display_draw"] == 0


def test_both_coprimary_contrasts_share_the_control(spec):
    comparisons = {c["name"]: c for c in spec["comparisons"]}
    assert set(comparisons) == {"mens_vs_control", "womens_vs_control"}
    assert {c["control_arm"] for c in comparisons.values()} == {"No E-Mail"}
    assert sorted(c["index"] for c in comparisons.values()) == [0, 1]


def test_outcome_hierarchy(spec):
    assert spec["outcomes"]["primary"] == "spend"
    assert spec["outcomes"]["secondary"] == ["visit", "conversion"]


def test_selection_uses_only_pretreatment_covariates(spec):
    selection = spec["selection"]
    outcomes = {"spend", "visit", "conversion", "segment"}
    assert not outcomes & set(selection["selection_covariates"])
    assert selection["intercept_target_mean_propensity"] == 0.5
    signs = {t["name"]: t["coefficient"] for t in selection["terms"]}
    assert signs == {"z_log_history": 0.75, "z_recency": -0.75, "multichannel": 0.5, "newbie": -0.5}


def test_scenarios(spec):
    scenarios = spec["scenarios"]
    assert scenarios["A"]["selection_strength_k"] == 1.0 and scenarios["A"]["hidden_covariates"] == []
    assert scenarios["B"]["selection_strength_k"] == 2.0 and scenarios["B"]["hidden_covariates"] == []
    assert scenarios["C"]["selection_strength_k"] == 1.0 and scenarios["C"]["hidden_covariates"] == ["recency"]
    # The hidden covariate must be one that drives selection, otherwise Scenario C hides nothing.
    assert set(scenarios["C"]["hidden_covariates"]) <= set(spec["selection"]["selection_covariates"])


def test_analyst_covariates_exclude_outcomes_and_redundant_history_segment(spec):
    covariates = set(spec["analyst"]["covariates"])
    assert not covariates & {"spend", "visit", "conversion", "segment", "history_segment"}
    assert set(spec["selection"]["selection_covariates"]) <= covariates


def test_every_estimator_declares_an_estimand_with_a_matching_benchmark(spec):
    benchmarks = set(spec["benchmarks"])
    for estimator in spec["estimators"]:
        assert estimator["benchmark"] in benchmarks, estimator["name"]
        if estimator["estimand"] in {"ATE", "ATT"}:
            assert estimator["estimand"] == estimator["benchmark"], estimator["name"]
    matching = next(e for e in spec["estimators"] if e["name"] == "matching_nn")
    assert matching["estimand"] == "ATT" and matching["benchmark"] == "ATT"
