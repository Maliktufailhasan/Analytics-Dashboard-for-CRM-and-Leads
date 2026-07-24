"""
Central configuration for the CRM Analytics Dashboard.

Keeping constants in one place makes the project easier to tweak.
If you want a bigger dataset, a different color palette, or a new list
of industries, edit this file — not the code that uses these values.
"""

from pathlib import Path

# --- Project paths ----------------------------------------------------------
# We derive every path from the project root so the app works no matter
# where the repo is cloned on disk.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
ASSETS_DIR = PROJECT_ROOT / "assets"

# The synthetic dataset is written here on first run and reused afterwards.
DATA_FILE = DATA_DIR / "crm_dataset.csv"

# --- Dataset generation -----------------------------------------------------
# The number of rows in the generated CRM dataset.
NUM_ROWS = 5000

# A fixed random seed keeps generated data (and model splits) reproducible.
RANDOM_SEED = 42

# The categorical vocabularies used both for data generation and for
# building dropdown filters in the dashboard.
INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Retail", "Manufacturing",
    "Education", "Real Estate", "Telecom", "Energy", "Hospitality",
]

COUNTRIES = [
    "United States", "United Kingdom", "Germany", "France", "Canada",
    "Australia", "India", "Brazil", "Japan", "United Arab Emirates",
]

LEAD_SOURCES = [
    "Website", "Referral", "Cold Call", "Email Campaign",
    "Social Media", "Trade Show", "Partner", "Paid Ads",
]

COMPANY_SIZES = ["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"]

SALESPEOPLE = [
    "Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson",
    "Emma Brown", "Frank Miller", "Grace Lee", "Henry Chen",
]

LEAD_STATUSES = [
    "New", "Contacted", "Qualified", "Proposal Sent",
    "Negotiation", "Won", "Lost",
]

# --- UI ---------------------------------------------------------------------
# Consistent Plotly color palette used across every chart.
CHART_COLORS = [
    "#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

# Streamlit page configuration values.
APP_TITLE = "CRM Analytics Dashboard"
APP_ICON = "📊"
