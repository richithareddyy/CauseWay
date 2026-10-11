"""Numeric encoding of the pre-treatment covariates.

One encoding is shared by balance diagnostics, regression adjustment and the observational benchmark:
ln(history), recency (linear), mens, womens, newbie, zip_code dummies (reference Suburban) and channel
dummies (reference Phone). Outcomes and the treatment are never part of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COVARIATE_COLUMNS = [
    "ln_history",
    "recency",
    "mens",
    "womens",
    "newbie",
    "zip_urban",
    "zip_rural",
    "channel_web",
    "channel_multichannel",
]

COVARIATE_LABELS = {
    "ln_history": "ln(past-year spend)",
    "recency": "Months since last purchase",
    "mens": "Bought Mens merchandise",
    "womens": "Bought Womens merchandise",
    "newbie": "New customer (past 12 months)",
    "zip_urban": "Urban ZIP code",
    "zip_rural": "Rural ZIP code",
    "channel_web": "Purchased via Web only",
    "channel_multichannel": "Purchased via multiple channels",
}


def covariate_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode covariates from a frame with canonical labels (zip_code uses 'Suburban')."""
    zip_code = frame["zip_code"].astype(str)
    channel = frame["channel"].astype(str)
    return pd.DataFrame(
        {
            "ln_history": np.log(frame["history"].astype(float)),
            "recency": frame["recency"].astype(float),
            "mens": frame["mens"].astype(float),
            "womens": frame["womens"].astype(float),
            "newbie": frame["newbie"].astype(float),
            "zip_urban": (zip_code == "Urban").astype(float),
            "zip_rural": (zip_code == "Rural").astype(float),
            "channel_web": (channel == "Web").astype(float),
            "channel_multichannel": (channel == "Multichannel").astype(float),
        },
        index=frame.index,
    )
