"""
Sales forecasting with Linear Regression.

We aggregate historical revenue into a monthly time series and fit a very
simple linear-regression trend model. This is intentionally basic — it is
easier to explain than ARIMA/Prophet and it demonstrates the workflow.

The model takes a single feature: `month_index` (0 for the first month in
the dataset, 1 for the next, ...) and predicts total monthly revenue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from models._storage import load_bundle, save_bundle


MODEL_NAME = "sales_forecast"


@dataclass
class ForecastResult:
    model: LinearRegression
    monthly_history: pd.DataFrame  # columns: month, month_index, revenue
    mae: float
    r2: float
    last_month_index: int = 0
    last_month_date: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.today())


def _aggregate_monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Sum revenue by month using `Created_Date` as the timestamp."""
    working = df[["Created_Date", "Revenue"]].copy()
    working["Created_Date"] = pd.to_datetime(working["Created_Date"], errors="coerce")
    working = working.dropna(subset=["Created_Date"])
    # `to_period('M')` groups by calendar month regardless of the day.
    working["month"] = working["Created_Date"].dt.to_period("M").dt.to_timestamp()
    monthly = working.groupby("month", as_index=False)["Revenue"].sum()
    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["month_index"] = np.arange(len(monthly))
    return monthly


def train_sales_forecast(df: pd.DataFrame) -> ForecastResult:
    """Fit the linear-regression forecaster on aggregated monthly revenue."""
    monthly = _aggregate_monthly_revenue(df)
    if len(monthly) < 3:
        raise ValueError("Need at least 3 months of history to train the forecaster.")

    X = monthly[["month_index"]].values
    y = monthly["Revenue"].values

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    mae = float(mean_absolute_error(y, predictions))
    r2 = float(r2_score(y, predictions))

    return ForecastResult(
        model=model,
        monthly_history=monthly,
        mae=mae,
        r2=r2,
        last_month_index=int(monthly["month_index"].iloc[-1]),
        last_month_date=monthly["month"].iloc[-1],
    )


def save_forecast_model(result: ForecastResult) -> None:
    save_bundle(
        MODEL_NAME,
        {
            "model": result.model,
            "monthly_history": result.monthly_history,
            "mae": result.mae,
            "r2": result.r2,
            "last_month_index": result.last_month_index,
            "last_month_date": result.last_month_date,
        },
    )


def load_or_train(df: pd.DataFrame) -> dict[str, Any]:
    bundle = load_bundle(MODEL_NAME)
    if bundle is not None:
        return bundle
    result = train_sales_forecast(df)
    save_forecast_model(result)
    return load_bundle(MODEL_NAME)


def forecast_next_months(bundle: dict[str, Any], months_ahead: int = 6) -> pd.DataFrame:
    """Return a DataFrame with the predicted revenue for the next N months."""
    model: LinearRegression = bundle["model"]
    last_index: int = bundle["last_month_index"]
    last_date: pd.Timestamp = bundle["last_month_date"]

    future_indices = np.arange(last_index + 1, last_index + 1 + months_ahead)
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=months_ahead,
        freq="MS",
    )
    predictions = model.predict(future_indices.reshape(-1, 1))
    # Revenue can't be negative — clip small negative predictions to 0.
    predictions = np.clip(predictions, a_min=0, a_max=None)

    return pd.DataFrame({
        "month": future_dates,
        "month_index": future_indices,
        "Revenue": predictions,
        "type": "forecast",
    })
