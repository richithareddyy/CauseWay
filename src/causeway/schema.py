"""Expected structure of the Hillstrom file and the canonical encodings used downstream."""

from __future__ import annotations

import math

COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]
PRETREATMENT_COVARIATES = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]
TREATMENT = "segment"
OUTCOMES = ["visit", "conversion", "spend"]

BINARY_COLUMNS = ["mens", "womens", "newbie", "visit", "conversion"]
INTEGER_COLUMNS = ["recency", *BINARY_COLUMNS]
FLOAT_COLUMNS = ["history", "spend"]

RECENCY_RANGE = (1, 12)

CONTROL_ARM = "No E-Mail"
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
ARM_CODES = {"No E-Mail": "control", "Mens E-Mail": "mens", "Womens E-Mail": "womens"}

HISTORY_SEGMENTS = {
    "1) $0 - $100": (0.0, 100.0),
    "2) $100 - $200": (100.0, 200.0),
    "3) $200 - $350": (200.0, 350.0),
    "4) $350 - $500": (350.0, 500.0),
    "5) $500 - $750": (500.0, 750.0),
    "6) $750 - $1,000": (750.0, 1000.0),
    "7) $1,000 +": (1000.0, math.inf),
}

CHANNELS = ["Phone", "Web", "Multichannel"]
ZIP_CODES = ["Urban", "Suburban", "Rural"]

# Raw spellings that are recognised and normalised. Anything else is a validation failure, never a
# silent mapping.
ZIP_CODE_NORMALIZATION = {"Urban": "Urban", "Suburban": "Suburban", "Surburban": "Suburban", "Rural": "Rural"}

RAW_DOMAINS = {
    "segment": set(ARMS),
    "history_segment": set(HISTORY_SEGMENTS),
    "channel": set(CHANNELS),
    "zip_code": set(ZIP_CODE_NORMALIZATION),
}

# Pre-specified recency bands (docs/design.md, Section 2.6).
RECENCY_BANDS = {"1-3": (1, 3), "4-6": (4, 6), "7-9": (7, 9), "10-12": (10, 12)}
