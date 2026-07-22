"""
CRM Analytics Dashboard — Streamlit entry point.

Run this file with:
    streamlit run app.py

Everything else is broken up into small modules under `config/`, `data/`,
`preprocessing/`, `models/`, `charts/`, `utils/`, and `dashboard/`.
This file's only job is to:
  1. Configure the Streamlit page.
  2. Load the (cached) CRM data once.
  3. Route the user to the page they picked in the sidebar.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from dashboard import (
    customers_page,
    dashboard_page,
    leads_page,
    predictions_page,
    reports_page,
    sales_page,
    settings_page,
)
from data.loader import load_crm_data


# Map of navigation labels -> page render functions.
# Adding a new page is a two-step change: create a `dashboard/foo_page.py`
# module with a `render(df)` function, then add it to this dict.
PAGES = {
    "Dashboard":   dashboard_page.render,
    "Leads":       leads_page.render,
    "Sales":       sales_page.render,
    "Customers":   customers_page.render,
    "Predictions": predictions_page.render,
    "Reports":     reports_page.render,
    "Settings":    settings_page.render,
}


def _configure_page() -> None:
    """Apply Streamlit page config + inject the custom stylesheet."""
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon=settings.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load our tiny stylesheet — the file lives under `assets/`.
    css_path = settings.ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def main() -> None:
    _configure_page()

    # Load once; every page shares this DataFrame.
    try:
        df = load_crm_data()
    except Exception as exc:
        st.error(f"Failed to load CRM data: {exc}")
        st.stop()

    # --- Navigation -------------------------------------------------------
    st.sidebar.title(f"{settings.APP_ICON} {settings.APP_TITLE}")
    st.sidebar.caption("Educational CRM analytics + ML demo.")
    page_name = st.sidebar.radio("Navigation", list(PAGES.keys()), index=0)

    st.sidebar.divider()

    # Delegate to the chosen page. Wrapping the render call gives users a
    # friendly message instead of a stack trace when something breaks.
    try:
        PAGES[page_name](df)
    except Exception as exc:
        st.error(f"Something went wrong rendering the {page_name} page.")
        st.exception(exc)


if __name__ == "__main__":
    main()
