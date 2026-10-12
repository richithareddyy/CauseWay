"""Robustness checks for the randomized experiment (configs/robustness.toml)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from causeway import schema
from causeway.config import CONFIG_DIR
from causeway.experiment import (
    bca_interval,
    bootstrap_means,
    dim_acceleration,
    neyman_difference,
    stream_rng,
)

CONTRAST_ARMS = {
    "mens_vs_control": ("Mens E-Mail", "No E-Mail"),
    "womens_vs_control": ("Womens E-Mail", "No E-Mail"),
    "mens_vs_womens": ("Mens E-Mail", "Womens E-Mail"),
}


def load_robustness_plan(path: Path = CONFIG_DIR / "robustness.toml") -> dict:
    return tomllib.loads(path.read_text())


def spend_versions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """The pre-specified spend variants. None of them removes a customer."""
    spend = frame["spend"].astype(float)
    purchases = spend[spend > 0]
    top = purchases.max()
    below_top = purchases[purchases < top].max()
    return {
        "spend": spend,
        f"spend, ${top:.0f} recoded to ${below_top:.2f}": spend.where(spend < top, below_top),
        "spend winsorized at p99 of purchases": spend.clip(upper=purchases.quantile(0.99)),
        "spend winsorized at p95 of purchases": spend.clip(upper=purchases.quantile(0.95)),
        "log(1 + spend)": np.log1p(spend),
    }


def _effect_rows(frame: pd.DataFrame, outcomes: dict[str, pd.Series], label: str) -> list[dict]:
    rows = []
    for contrast, (treated_arm, control_arm) in CONTRAST_ARMS.items():
        treated = frame["segment"] == treated_arm
        control = frame["segment"] == control_arm
        for name, values in outcomes.items():
            effect = neyman_difference(values[treated].to_numpy(), values[control].to_numpy())
            rows.append(
                {
                    "analysis": label,
                    "contrast": contrast,
                    "outcome": name,
                    **effect.__dict__,
                    "n": int(treated.sum() + control.sum()),
                }
            )
    return rows


def experiment_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    """Alternative spend outcomes, and all outcomes again without exact duplicate rows."""
    rows = _effect_rows(frame, spend_versions(frame), "spend version")
    deduplicated = frame.drop_duplicates(subset=schema.COLUMNS, keep="first")
    outcomes = {name: deduplicated[name].astype(float) for name in ("spend", "visit", "conversion")}
    rows += _effect_rows(deduplicated, outcomes, f"without duplicate rows (n = {len(deduplicated):,})")
    return pd.DataFrame(rows)


def bootstrap_stability(frame: pd.DataFrame, plan: dict, n_resamples: int = 5000) -> pd.DataFrame:
    """BCa intervals for the spend effect under several bootstrap seeds."""
    rows = []
    for stream in plan["randomness"]["bootstrap_stability_streams"]:
        rng = stream_rng(plan["randomness"]["base_seed"], stream)
        arms = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
        values = {arm: frame.loc[frame["segment"] == arm, "spend"].to_numpy(dtype=float) for arm in arms}
        draws = dict(zip(arms, bootstrap_means([values[a] for a in arms], n_resamples, rng), strict=True))
        for contrast, (treated_arm, control_arm) in CONTRAST_ARMS.items():
            y_t, y_c = values[treated_arm], values[control_arm]
            estimate = y_t.mean() - y_c.mean()
            low, high = bca_interval(
                (draws[treated_arm] - draws[control_arm])[:, 0], estimate, dim_acceleration(y_t, y_c)
            )
            rows.append(
                {"stream": stream, "contrast": contrast, "estimate": estimate, "ci_low": low, "ci_high": high}
            )
    return pd.DataFrame(rows)
