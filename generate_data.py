"""
Generate a realistic synthetic CRM dataset.

Why synthetic data?
  - This is a portfolio/educational project so we cannot ship real customer data.
  - Generating data programmatically means anyone can run the app immediately
    without hunting for a CSV to download.

The generator injects intentional structure (e.g. bigger budgets are more
likely to convert) so that the machine learning models later on actually
have signal to learn from — otherwise every model would look useless.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import settings


def _random_dates(rng: np.random.Generator, n: int, start_days_ago: int = 730) -> pd.Series:
    """Return `n` random dates spread across the last `start_days_ago` days."""
    today = datetime.today()
    offsets = rng.integers(low=0, high=start_days_ago, size=n)
    dates = [today - timedelta(days=int(offset)) for offset in offsets]
    return pd.to_datetime(dates).normalize()


def generate_crm_dataset(num_rows: int = settings.NUM_ROWS, seed: int = settings.RANDOM_SEED) -> pd.DataFrame:
    """Build the synthetic CRM DataFrame used throughout the dashboard.

    Args:
        num_rows: How many leads/customers to generate.
        seed: Random seed so the same data is produced every run.

    Returns:
        A pandas DataFrame with all the fields described in the README.
    """
    rng = np.random.default_rng(seed)

    # --- Simple identity columns -------------------------------------------
    lead_ids = [f"L{100000 + i}" for i in range(num_rows)]
    customer_names = [f"Customer_{i:05d}" for i in range(num_rows)]

    # --- Categorical columns -----------------------------------------------
    industries = rng.choice(settings.INDUSTRIES, size=num_rows)
    countries = rng.choice(settings.COUNTRIES, size=num_rows)
    lead_sources = rng.choice(settings.LEAD_SOURCES, size=num_rows)
    company_sizes = rng.choice(settings.COMPANY_SIZES, size=num_rows)
    salespeople = rng.choice(settings.SALESPEOPLE, size=num_rows)

    # Weight statuses so most leads are still open and a smaller share
    # end up won/lost — this mirrors what a real sales pipeline looks like.
    status_weights = [0.20, 0.15, 0.15, 0.12, 0.10, 0.15, 0.13]
    statuses = rng.choice(settings.LEAD_STATUSES, size=num_rows, p=status_weights)

    # --- Numeric columns ---------------------------------------------------
    # Budgets are drawn from a lognormal so we get a realistic long tail
    # (a few big deals, many small ones).
    budgets = np.round(rng.lognormal(mean=9.5, sigma=0.7, size=num_rows), 2)
    meetings = rng.integers(low=0, high=20, size=num_rows)
    emails = rng.integers(low=0, high=50, size=num_rows)
    website_visits = rng.integers(low=0, high=100, size=num_rows)

    # --- Dates -------------------------------------------------------------
    created_dates = _random_dates(rng, num_rows, start_days_ago=730)
    # Last_Contact must fall after Created_Date, so we add a random gap.
    contact_gaps = rng.integers(low=0, high=180, size=num_rows)
    last_contacts = created_dates + pd.to_timedelta(contact_gaps, unit="D")

    # --- Target-like columns with intentional signal -----------------------
    # We want conversion probability to depend on things a model can learn:
    # bigger budget, more meetings, more emails => higher chance of converting.
    conversion_score = (
        (budgets / budgets.max()) * 0.5
        + (meetings / meetings.max()) * 0.3
        + (emails / emails.max()) * 0.2
    )
    # Add noise so the relationship isn't trivial to learn perfectly.
    conversion_score = np.clip(conversion_score + rng.normal(0, 0.15, num_rows), 0, 1)
    converted = (conversion_score > 0.55).astype(int)

    # Revenue exists only for converted leads; the rest are zero.
    revenue = np.where(
        converted == 1,
        np.round(budgets * rng.uniform(0.6, 1.3, size=num_rows), 2),
        0.0,
    )

    # Churn only makes sense for customers we actually converted.
    # We model churn as more likely when engagement (meetings/emails) is low.
    # Threshold tuned so ~25-30% of converted customers churn — a realistic rate.
    engagement = (meetings / meetings.max()) * 0.5 + (emails / emails.max()) * 0.5
    churn_prob = np.clip(0.9 - engagement + rng.normal(0, 0.15, num_rows), 0, 1)
    churned = np.where(converted == 1, (churn_prob > 0.5).astype(int), 0)

    # --- Assemble the DataFrame -------------------------------------------
    df = pd.DataFrame({
        "Lead_ID": lead_ids,
        "Customer_Name": customer_names,
        "Industry": industries,
        "Country": countries,
        "Lead_Source": lead_sources,
        "Budget": budgets,
        "Company_Size": company_sizes,
        "Salesperson": salespeople,
        "Status": statuses,
        "Revenue": revenue,
        "Meetings": meetings,
        "Emails": emails,
        "Website_Visits": website_visits,
        "Last_Contact": last_contacts,
        "Created_Date": created_dates,
        "Converted": converted,
        "Churned": churned,
    })

    return df


def save_dataset(df: pd.DataFrame, path=None) -> None:
    """Persist the dataset to disk as CSV."""
    if path is None:
        path = settings.DATA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # Allow running `python -m data.generate_data` to rebuild the CSV.
    dataset = generate_crm_dataset()
    save_dataset(dataset)
    print(f"Saved {len(dataset):,} rows to {settings.DATA_FILE}")
