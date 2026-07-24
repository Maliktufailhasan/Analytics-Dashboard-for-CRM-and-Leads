"""
Plotly chart builders used across the dashboard.

Each function takes a DataFrame (or two) and returns a plotly.graph_objects
Figure. Keeping the chart code out of the page files means:
  - The Streamlit pages read like a story (get data -> build chart -> show it).
  - We can reuse the same chart in multiple pages if needed.
  - The look-and-feel stays consistent (same palette, same layout).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import settings


# A single place to configure common layout options.
def _apply_common_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


# ---------------------------------------------------------------------- line
def revenue_over_time_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart of daily revenue (summed) over `Created_Date`."""
    daily = (
        df.assign(day=pd.to_datetime(df["Created_Date"]).dt.to_period("D").dt.to_timestamp())
          .groupby("day", as_index=False)["Revenue"].sum()
    )
    fig = px.line(
        daily, x="day", y="Revenue",
        color_discrete_sequence=[settings.CHART_COLORS[0]],
    )
    fig.update_traces(mode="lines")
    return _apply_common_layout(fig, title="Revenue Over Time")


# ---------------------------------------------------------------------- bar
def leads_by_source_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of leads grouped by acquisition source."""
    counts = df["Lead_Source"].value_counts().reset_index()
    counts.columns = ["Lead_Source", "Leads"]
    fig = px.bar(
        counts, x="Lead_Source", y="Leads",
        color="Lead_Source", color_discrete_sequence=settings.CHART_COLORS,
    )
    fig.update_layout(showlegend=False)
    return _apply_common_layout(fig, title="Leads by Source")


def revenue_by_salesperson_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of total revenue attributed to each salesperson."""
    totals = df.groupby("Salesperson", as_index=False)["Revenue"].sum()
    totals = totals.sort_values("Revenue", ascending=False)
    fig = px.bar(
        totals, x="Salesperson", y="Revenue",
        color="Salesperson", color_discrete_sequence=settings.CHART_COLORS,
    )
    fig.update_layout(showlegend=False)
    return _apply_common_layout(fig, title="Revenue by Salesperson")


# ---------------------------------------------------------------------- pie
def status_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Pie chart showing the current mix of lead statuses."""
    counts = df["Status"].value_counts().reset_index()
    counts.columns = ["Status", "Leads"]
    fig = px.pie(
        counts, names="Status", values="Leads", hole=0.45,
        color_discrete_sequence=settings.CHART_COLORS,
    )
    return _apply_common_layout(fig, title="Lead Status Breakdown")


# --------------------------------------------------------------------- funnel
def sales_funnel_chart(df: pd.DataFrame) -> go.Figure:
    """Funnel chart of lead volume at each pipeline stage."""
    # Preserve pipeline order rather than alphabetical.
    order = ["New", "Contacted", "Qualified", "Proposal Sent", "Negotiation", "Won"]
    counts = df["Status"].value_counts().reindex(order).fillna(0)
    fig = go.Figure(go.Funnel(
        y=order,
        x=counts.values,
        marker=dict(color=settings.CHART_COLORS[: len(order)]),
        textinfo="value+percent initial",
    ))
    return _apply_common_layout(fig, title="Sales Funnel")


# -------------------------------------------------------------------- scatter
def budget_vs_revenue_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of budget vs. actual revenue, colored by conversion."""
    fig = px.scatter(
        df,
        x="Budget",
        y="Revenue",
        color=df["Converted"].map({0: "Not Converted", 1: "Converted"}),
        color_discrete_sequence=[settings.CHART_COLORS[3], settings.CHART_COLORS[2]],
        hover_data=["Industry", "Country", "Salesperson"],
        opacity=0.6,
    )
    fig.update_layout(legend_title_text="")
    return _apply_common_layout(fig, title="Budget vs. Revenue")


# ------------------------------------------------------------------- histogram
def budget_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of the budget column — good for spotting outliers."""
    fig = px.histogram(
        df, x="Budget", nbins=40,
        color_discrete_sequence=[settings.CHART_COLORS[0]],
    )
    return _apply_common_layout(fig, title="Budget Distribution")


# --------------------------------------------------------------------- heatmap
def conversion_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of conversion rate by (Country, Industry)."""
    pivot = df.pivot_table(
        index="Country", columns="Industry", values="Converted", aggfunc="mean"
    ).fillna(0)
    fig = px.imshow(
        pivot,
        text_auto=".0%",
        color_continuous_scale="Blues",
        aspect="auto",
    )
    return _apply_common_layout(fig, title="Conversion Rate by Country x Industry")


# --------------------------------------------------------------------- treemap
def revenue_treemap(df: pd.DataFrame) -> go.Figure:
    """Treemap of revenue split by Country -> Industry."""
    grouped = df.groupby(["Country", "Industry"], as_index=False)["Revenue"].sum()
    grouped = grouped[grouped["Revenue"] > 0]
    fig = px.treemap(
        grouped,
        path=["Country", "Industry"],
        values="Revenue",
        color="Revenue",
        color_continuous_scale="Blues",
    )
    return _apply_common_layout(fig, title="Revenue Distribution")


# ----------------------------------------------------------------------- gauge
def conversion_gauge(value_percent: float, title: str = "Conversion Rate") -> go.Figure:
    """A single-metric gauge, useful for headline KPIs."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_percent,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": settings.CHART_COLORS[0]},
            "steps": [
                {"range": [0, 33], "color": "#F8D7DA"},
                {"range": [33, 66], "color": "#FFF3CD"},
                {"range": [66, 100], "color": "#D4EDDA"},
            ],
        },
        title={"text": title},
    ))
    return _apply_common_layout(fig)


# ------------------------------------------------------------------- forecast
def sales_forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Historical monthly revenue in solid line, forecast in dashed line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["month"], y=history["Revenue"],
        mode="lines+markers", name="History",
        line=dict(color=settings.CHART_COLORS[0]),
    ))
    fig.add_trace(go.Scatter(
        x=forecast["month"], y=forecast["Revenue"],
        mode="lines+markers", name="Forecast",
        line=dict(color=settings.CHART_COLORS[1], dash="dash"),
    ))
    return _apply_common_layout(fig, title="Monthly Revenue: History + Forecast")


# --------------------------------------------------------- feature importance
def feature_importance_chart(importances: dict[str, float], top_n: int = 10) -> go.Figure:
    """Horizontal bar chart showing the most important model features."""
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    labels = [name for name, _ in items][::-1]  # reverse so biggest is on top
    values = [value for _, value in items][::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=settings.CHART_COLORS[0]),
    ))
    return _apply_common_layout(fig, title="Top Feature Importances")


# ------------------------------------------------------------- segmentation
def segment_scatter(df_with_segments: pd.DataFrame) -> go.Figure:
    """Scatter plot of Budget vs Total_Engagement colored by segment."""
    fig = px.scatter(
        df_with_segments,
        x="Budget",
        y="Total_Engagement",
        color="Segment",
        color_discrete_sequence=settings.CHART_COLORS,
        opacity=0.6,
        hover_data=["Industry", "Country"],
    )
    return _apply_common_layout(fig, title="Customer Segments")
