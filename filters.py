"""
Shared Streamlit sidebar filters.

Every dashboard page can call `render_sidebar_filters(df)` to get the same
set of filter widgets and the resulting filtered DataFrame. Centralising
this avoids drift between pages ("why does the date filter behave
differently on the Sales page?").
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st


def _multiselect_all(label: str, options: list[str], key: str) -> list[str]:
    """A multiselect where selecting nothing means 'all values'."""
    selected = st.sidebar.multiselect(label, options=options, default=[], key=key)
    return selected if selected else options


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Draw sidebar filters, return a copy of `df` filtered by user choices."""
    st.sidebar.header("Filters")

    # --- Date range ---------------------------------------------------------
    min_date = df["Created_Date"].min().date()
    max_date = df["Created_Date"].max().date()
    default_start = max(min_date, max_date - timedelta(days=365))
    date_range = st.sidebar.date_input(
        "Created between",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
        key="filter_dates",
    )
    # `date_input` returns a single date if the user only picked one — guard for it.
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, max_date

    # --- Categorical filters ----------------------------------------------
    countries = _multiselect_all("Country", sorted(df["Country"].unique()), "filter_country")
    industries = _multiselect_all("Industry", sorted(df["Industry"].unique()), "filter_industry")
    sources = _multiselect_all("Lead Source", sorted(df["Lead_Source"].unique()), "filter_source")
    salespeople = _multiselect_all("Salesperson", sorted(df["Salesperson"].unique()), "filter_salesperson")
    statuses = _multiselect_all("Lead Status", sorted(df["Status"].unique()), "filter_status")

    # --- Apply filters -----------------------------------------------------
    mask = (
        (df["Created_Date"].dt.date >= start_date)
        & (df["Created_Date"].dt.date <= end_date)
        & df["Country"].isin(countries)
        & df["Industry"].isin(industries)
        & df["Lead_Source"].isin(sources)
        & df["Salesperson"].isin(salespeople)
        & df["Status"].isin(statuses)
    )
    filtered = df.loc[mask].copy()

    st.sidebar.caption(f"Showing **{len(filtered):,}** of {len(df):,} rows")
    return filtered
