"""
KPI computations for the dashboard.

Each function accepts a (filtered) DataFrame and returns a single number
or a small dict. Keeping them tiny and pure makes them easy to test and
easy to re-use across pages.
"""

from __future__ import annotations

import pandas as pd


def total_leads(df: pd.DataFrame) -> int:
    return int(len(df))


def converted_leads(df: pd.DataFrame) -> int:
    return int(df["Converted"].sum())


def conversion_rate(df: pd.DataFrame) -> float:
    """Return conversion rate as a percentage (0-100)."""
    if len(df) == 0:
        return 0.0
    return float(df["Converted"].mean() * 100)


def total_revenue(df: pd.DataFrame) -> float:
    return float(df["Revenue"].sum())


def average_deal_size(df: pd.DataFrame) -> float:
    """Average revenue among won deals only (empty pipeline returns 0)."""
    won = df[df["Revenue"] > 0]
    if len(won) == 0:
        return 0.0
    return float(won["Revenue"].mean())


def active_customers(df: pd.DataFrame) -> int:
    """Converted customers who haven't churned yet."""
    return int(((df["Converted"] == 1) & (df["Churned"] == 0)).sum())


def churn_rate(df: pd.DataFrame) -> float:
    """Churn rate (0-100) among converted customers."""
    customers = df[df["Converted"] == 1]
    if len(customers) == 0:
        return 0.0
    return float(customers["Churned"].mean() * 100)


def format_currency(value: float) -> str:
    """Render a float as a compact currency string (e.g. `$1.2M`)."""
    if value >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    if value >= 1_000:
        return f"${value/1_000:,.1f}K"
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:,.1f}%"


def format_number(value: float) -> str:
    return f"{value:,.0f}"
