"""Greedy 1:1 nearest-neighbour matching on the logit propensity score, without replacement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats


@dataclass
class MatchResult:
    control_for_treated: np.ndarray  # index into the control arrays, or -1 when unmatched
    distances: np.ndarray  # logit-scale distance for each matched pair
    caliper: float


def pooled_logit_sd(scores: np.ndarray, treated: np.ndarray) -> float:
    """Pooled SD of the logit propensity, sqrt((var_t + var_c) / 2) (Austin, 2011)."""
    logit = special.logit(scores)
    return float(np.sqrt((logit[treated].var(ddof=1) + logit[~treated].var(ddof=1)) / 2))


def _find(parent: list[int], i: int) -> int:
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def greedy_match(
    treated_logit: np.ndarray,
    control_logit: np.ndarray,
    treated_order: np.ndarray,
    control_row_ids: np.ndarray,
    caliper: float | None,
) -> MatchResult:
    """Nearest available control for each treated unit, in the given order.

    Union-find over the sorted controls keeps each lookup close to constant time.
    """
    order = np.lexsort((control_row_ids, control_logit))
    values = control_logit[order]
    row_ids = control_row_ids[order]
    n = len(values)
    # Sentinels: index 0 = "nothing to the left", index n + 1 = "nothing to the right".
    left = list(range(n + 2))
    right = list(range(n + 2))
    matched = np.full(len(treated_logit), -1, dtype=np.int64)
    distances = []
    for i in treated_order:
        position = int(np.searchsorted(values, treated_logit[i]))
        candidates = []
        lj = _find(left, position)  # position is 1-based in the union-find arrays
        if lj >= 1:
            candidates.append(lj - 1)
        rj = _find(right, position + 1)
        if rj <= n:
            candidates.append(rj - 1)
        if not candidates:
            break
        best = min(candidates, key=lambda j: (abs(values[j] - treated_logit[i]), row_ids[j]))
        distance = abs(values[best] - treated_logit[i])
        if caliper is not None and distance > caliper:
            continue
        matched[i] = order[best]
        distances.append(distance)
        left[best + 1] = best  # mark used: point past it in both directions
        right[best + 1] = best + 2
    return MatchResult(matched, np.asarray(distances), caliper if caliper is not None else np.inf)


def match_on_propensity(
    scores: np.ndarray, treated: np.ndarray, row_ids: np.ndarray, caliper_sd: float | None
) -> MatchResult:
    """Apply the frozen matching rules to one sample; returns indices into the control subset."""
    logit = special.logit(scores)
    t_logit, c_logit = logit[treated], logit[~treated]
    t_scores, t_rows = scores[treated], row_ids[treated]
    treated_order = np.lexsort((t_rows, -t_scores))
    caliper = None if caliper_sd is None else caliper_sd * pooled_logit_sd(scores, treated)
    return greedy_match(t_logit, c_logit, treated_order, row_ids[~treated], caliper)


def matched_att(y: np.ndarray, treated: np.ndarray, match: MatchResult, ci_level: float = 0.95) -> dict:
    """ATT among matched treated units with a paired-difference SE conditional on the matches."""
    y_t, y_c = np.asarray(y, dtype=float)[treated], np.asarray(y, dtype=float)[~treated]
    has_match = match.control_for_treated >= 0
    differences = y_t[has_match] - y_c[match.control_for_treated[has_match]]
    n_pairs = len(differences)
    estimate = float(differences.mean()) if n_pairs else np.nan
    se = float(differences.std(ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else np.nan
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "n_pairs": n_pairs,
        "share_treated_discarded": float(1 - has_match.mean()),
    }


def matching_weights(treated: np.ndarray, match: MatchResult) -> np.ndarray:
    """Represent the matched sample as 0/1 weights on the original sample (for balance diagnostics)."""
    weights = np.zeros(len(treated))
    treated_index, control_index = np.flatnonzero(treated), np.flatnonzero(~treated)
    has_match = match.control_for_treated >= 0
    weights[treated_index[has_match]] = 1.0
    weights[control_index[match.control_for_treated[has_match]]] = 1.0
    return weights
