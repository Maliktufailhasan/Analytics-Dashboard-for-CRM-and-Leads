"""
Load the CRM dataset from disk (or generate it on first run).

The Streamlit app calls `load_crm_data()` from every page, so we cache the
result with `st.cache_data` to avoid re-reading the CSV on every rerun.
"""

import pandas as pd
import streamlit as st

from config import settings
from data.generate_data import generate_crm_dataset, save_dataset


@st.cache_data(show_spinner="Loading CRM data...")
def load_crm_data() -> pd.DataFrame:
    """Return the CRM DataFrame, generating and saving it on first call.

    Streamlit caches the result so repeated page navigations are fast.
    """
    # If the CSV doesn't exist yet, build it once and persist it.
    if not settings.DATA_FILE.exists():
        df = generate_crm_dataset()
        save_dataset(df)
    else:
        df = pd.read_csv(settings.DATA_FILE)

    # Convert date columns to real datetimes — the CSV round-trip turns
    # them into plain strings which breaks filtering and charting.
    for date_col in ("Created_Date", "Last_Contact"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    return df


def refresh_crm_data() -> pd.DataFrame:
    """Force-regenerate the dataset and clear Streamlit's cache."""
    df = generate_crm_dataset()
    save_dataset(df)
    load_crm_data.clear()
    return load_crm_data()
