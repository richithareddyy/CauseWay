"""Standardized mean differences against hand-computed values."""

import math

import numpy as np
import pandas as pd
import pytest

from causeway.balance import balance_table, standardized_mean_difference


def test_smd_matches_hand_calculation():
    # means 2 and 1; sample variances 1 and 1 -> pooled SD 1 -> SMD 1
    assert standardized_mean_difference([1, 2, 3], [0, 1, 2]) == pytest.approx(1.0)


def test_smd_uses_pooled_sample_variances():
    treated, control = np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1])
    pooled = math.sqrt((np.var(treated, ddof=1) + np.var(control, ddof=1)) / 2)
    assert standardized_mean_difference(treated, control) == pytest.approx(0.25 / pooled)


def test_smd_is_antisymmetric():
    a, b = [3.0, 5.0, 9.0], [1.0, 2.0, 2.5]
    assert standardized_mean_difference(a, b) == pytest.approx(-standardized_mean_difference(b, a))


def test_weighted_smd_keeps_the_unweighted_scale():
    treated, control = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    unweighted = standardized_mean_difference(treated, control)
    weighted = standardized_mean_difference(treated, control, treated_weights=np.array([1.0, 3.0]))
    pooled = math.sqrt(0.5)
    assert unweighted == 0.0
    assert weighted == pytest.approx((0.75 - 0.5) / pooled)


def test_constant_groups():
    assert standardized_mean_difference([1, 1], [1, 1]) == 0.0
    assert standardized_mean_difference([2, 2], [1, 1]) == math.inf


def test_balance_table():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 0.0, 1.0, 2.0]})
    table = balance_table(frame, pd.Series([1, 1, 1, 0, 0, 0]), ["x"])
    row = table.iloc[0]
    assert (row["treated_mean"], row["control_mean"]) == (2.0, 1.0)
    assert row["smd"] == pytest.approx(1.0)
