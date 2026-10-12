"""Descriptive analysis of the randomized experiment (no causal estimation here)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causeway import schema
from causeway.features import covariate_matrix

OUTCOMES = ["spend", "visit", "conversion"]


def outcome_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Outcome rates and the spend distribution by arm. Spend among converters is descriptive only."""
    rows = []
    for arm in schema.ARMS:
        group = frame[frame["segment"] == arm]
        converters = group.loc[group["conversion"] == 1, "spend"]
        rows.append(
            {
                "arm": arm,
                "n": len(group),
                "visit_rate": group["visit"].mean(),
                "conversion_rate": group["conversion"].mean(),
                "spend_mean": group["spend"].mean(),
                "spend_sd": group["spend"].std(ddof=1),
                "share_zero_spend": (group["spend"] == 0).mean(),
                "n_converters": len(converters),
                "spend_given_conversion_mean": converters.mean(),
                "spend_given_conversion_median": converters.median(),
                "spend_given_conversion_p90": converters.quantile(0.9),
                "spend_given_conversion_max": converters.max(),
            }
        )
    return pd.DataFrame(rows)


def spend_boundary(frame: pd.DataFrame, bins: list[list[float]]) -> dict:
    """Upper tail of spend around the observed maximum."""
    positive = frame.loc[frame["spend"] > 0]
    maximum = float(positive["spend"].max())
    at_max = positive[positive["spend"] == maximum]
    bands = []
    for low, high in [*bins, [maximum, maximum]]:
        in_band = positive[(positive["spend"] >= low) & (positive["spend"] < high)] if low < high else at_max
        bands.append(
            {
                "band": f"[{low:.2f}, {high:.2f})" if low < high else f"= {maximum:.2f}",
                "n": len(in_band),
                "share_of_converters": len(in_band) / len(positive),
                **{f"n_{schema.ARM_CODES[a]}": int((in_band["segment"] == a).sum()) for a in schema.ARMS},
            }
        )
    top_values = positive["spend"].round(2).value_counts().head(10)
    distinct_below_max = np.sort(positive.loc[positive["spend"] < maximum, "spend"].unique())[-10:]
    return {
        "maximum": maximum,
        "n_at_maximum": len(at_max),
        "n_converters": len(positive),
        "share_at_maximum": len(at_max) / len(positive),
        "bands": bands,
        "most_frequent_values": [{"spend": float(v), "n": int(n)} for v, n in top_values.items()],
        "largest_values_below_maximum": [float(v) for v in distinct_below_max],
        "converters_at_maximum_by_arm": {
            a: {
                "n_at_maximum": int((at_max["segment"] == a).sum()),
                "n_converters": int((positive["segment"] == a).sum()),
            }
            for a in schema.ARMS
        },
        "quantiles_given_conversion": {
            str(q): float(positive["spend"].quantile(q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
    }


def covariate_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations among the encoded pre-treatment covariates."""
    return covariate_matrix(frame).corr()


def prognostic_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each covariate with each outcome among controls."""
    control = frame[frame["segment"] == schema.CONTROL_ARM]
    covariates = covariate_matrix(control)
    return pd.DataFrame(
        {outcome: covariates.corrwith(control[outcome].astype(float)) for outcome in OUTCOMES}
    )
