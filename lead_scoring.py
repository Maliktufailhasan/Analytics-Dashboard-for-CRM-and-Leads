"""
Lead scoring.

Every lead is scored 0-100 based on transparent, weighted signals. We use
a rules-based score (not an ML model) so a salesperson can literally read
the weight table and understand why a lead is "hot" or "cold".

Signals used:
  - Budget         (larger budget => higher score)
  - Meetings       (more meetings => stronger interest)
  - Emails         (more email interactions => engagement)
  - Website_Visits (visits are a self-serve signal)
  - Status         (later pipeline stages score higher)

Weights are chosen to sum to 100.
"""

from __future__ import annotations

import pandas as pd


# Score weights (must total 100).
WEIGHTS = {
    "Budget": 35,
    "Meetings": 20,
    "Emails": 10,
    "Website_Visits": 10,
    "Status": 25,
}

# Score value each status contributes to the "Status" component.
# Later stages of the pipeline are worth more.
STATUS_SCORES = {
    "New": 10,
    "Contacted": 25,
    "Qualified": 50,
    "Proposal Sent": 70,
    "Negotiation": 85,
    "Won": 100,
    "Lost": 0,
}


def _min_max_scale(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0..100. Safe for constant series."""
    lo = series.min()
    hi = series.max()
    if hi == lo:
        return pd.Series([50.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo) * 100.0


def compute_lead_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with two extra columns: `Lead_Score` and `Lead_Tier`.

    The score is a weighted sum of scaled signals; the tier bucket
    (Hot/Warm/Cold) is a friendly label sales teams can filter on.
    """
    scored = df.copy()

    # Scale each numeric signal to 0..100 so weights are on a common scale.
    budget_score = _min_max_scale(scored["Budget"])
    meetings_score = _min_max_scale(scored["Meetings"])
    emails_score = _min_max_scale(scored["Emails"])
    visits_score = _min_max_scale(scored["Website_Visits"])

    # Map status to its predefined score.
    status_score = scored["Status"].map(STATUS_SCORES).fillna(0)

    total_weight = sum(WEIGHTS.values())  # should be 100

    scored["Lead_Score"] = (
        budget_score * WEIGHTS["Budget"]
        + meetings_score * WEIGHTS["Meetings"]
        + emails_score * WEIGHTS["Emails"]
        + visits_score * WEIGHTS["Website_Visits"]
        + status_score * WEIGHTS["Status"]
    ) / total_weight

    scored["Lead_Score"] = scored["Lead_Score"].round(1)
    scored["Lead_Tier"] = pd.cut(
        scored["Lead_Score"],
        bins=[-1, 39, 69, 100],
        labels=["Cold", "Warm", "Hot"],
    )

    return scored


def score_single_lead(row: pd.Series, reference_df: pd.DataFrame) -> tuple[float, str]:
    """Return `(score, tier)` for a single lead using min/max from a reference set.

    The reference DataFrame supplies the min/max values used for scaling so
    that a lead is scored on the same scale as the rest of the pipeline.
    """
    df = pd.concat([reference_df, row.to_frame().T], ignore_index=True)
    scored = compute_lead_scores(df)
    last = scored.iloc[-1]
    return float(last["Lead_Score"]), str(last["Lead_Tier"])
