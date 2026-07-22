# CRM Analytics Dashboard

An educational, end-to-end CRM analytics dashboard built entirely with
Python and Streamlit. It generates a realistic synthetic CRM dataset,
trains a small suite of machine-learning models on it, and exposes
everything through a friendly multi-page Streamlit UI.

The code is deliberately written to be **readable first, clever second**.
It is intended as portfolio / interview / classroom material.

---

## Project overview

The app has seven pages:

| Page          | What it does                                                              |
|---------------|---------------------------------------------------------------------------|
| Dashboard     | Executive KPIs + overview charts.                                         |
| Leads         | Lead volume, sources, and transparent 0-100 lead scoring.                 |
| Sales         | Revenue, deal size, salesperson leaderboard, sales forecast.              |
| Customers     | Segmentation and churn view of converted customers.                       |
| Predictions   | Pick any lead and see conversion / churn / score / segment predictions.   |
| Reports       | Filter + export to CSV / Excel, plus a Markdown summary report.           |
| Settings      | Regenerate the dataset, retrain models, clear caches.                     |

---

## Folder structure

```
.
├── app.py                     # Streamlit entry point
├── requirements.txt
├── README.md
├── config/
│   └── settings.py            # Paths, constants, palette, dataset options
├── data/
│   ├── generate_data.py       # Synthetic CRM dataset generator (~5k rows)
│   └── loader.py              # Cached CSV loader
├── preprocessing/
│   └── pipeline.py            # Reusable fit/transform pipeline
├── models/
│   ├── _storage.py            # Joblib save/load helpers
│   ├── lead_conversion.py     # Random Forest classifier
│   ├── churn_prediction.py    # Random Forest classifier
│   ├── sales_forecast.py      # Linear Regression on monthly revenue
│   ├── segmentation.py        # K-Means clustering
│   ├── lead_scoring.py        # Transparent rules-based score
│   └── saved/                 # Trained .joblib bundles land here
├── dashboard/
│   ├── _components.py         # Shared KPI/section helpers
│   ├── dashboard_page.py
│   ├── leads_page.py
│   ├── sales_page.py
│   ├── customers_page.py
│   ├── predictions_page.py
│   ├── reports_page.py
│   └── settings_page.py
├── charts/
│   └── plotly_charts.py       # All Plotly figure builders
├── utils/
│   ├── filters.py             # Shared sidebar filters
│   └── kpis.py                # KPI calculations + formatting
└── assets/
    └── styles.css             # A small custom stylesheet
```

Every file has one single responsibility. If you're looking for a bug, the
folder name tells you where to go.

---

## Installation

1. **Clone the repo** (or download the folder).
2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

Requires Python 3.10+.

---

## Running the application

From the project root:

```bash
streamlit run app.py
```

On first launch the app will:

1. Generate `data/crm_dataset.csv` (~5,000 rows).
2. Train and cache the ML models under `models/saved/` the first time each
   page that needs them is opened.

Both steps only happen once — subsequent launches are instant.

To regenerate the data or retrain models, use the **Settings** page.

---

## Module walkthrough

### `config/settings.py`
Central place for paths, dataset size, random seed, chart palette, and the
categorical vocabularies used both by the data generator and the sidebar
filters.

### `data/generate_data.py`
Builds a realistic synthetic CRM dataset. It injects real signal (bigger
budgets are more likely to convert, low-engagement customers are more
likely to churn) so the models later on actually learn something.

### `data/loader.py`
Loads the CSV with `st.cache_data` so page navigation stays snappy.
Regenerates the CSV automatically on first run.

### `preprocessing/pipeline.py`
The reusable **`CRMPreprocessor`** class. Handles:

- Duplicate removal.
- Numeric imputation with training-set medians.
- Categorical label encoding with an `"Unknown"` sentinel for unseen values.
- Standard-scaler feature scaling.
- Simple feature engineering (`Total_Engagement`, `Budget_Per_Meeting`,
  `Days_Since_Created`).

Same object is used at training time **and** at prediction time — this is
the single most important defense against subtle bugs in ML pipelines.

### `models/`
Every model lives in its own file with a consistent shape:

- `train_*()` — fit the model, return a dataclass with metrics.
- `save_*()` / `load_or_train()` — persist and lazy-load via joblib.
- `predict_*()` — score a single lead / row.

### `charts/plotly_charts.py`
Every chart the app uses is here. Pages import these builders and simply
pass them the (filtered) DataFrame — the pages themselves stay short and
declarative.

### `utils/filters.py` and `utils/kpis.py`
Shared sidebar filter widgets and KPI math. Keeping this in one place
means every page filters and calculates the same way.

### `dashboard/*_page.py`
One `render(df)` function per page. Navigation is a plain `st.sidebar.radio`
that maps labels to render functions in `app.py`.

---

## Machine learning models

### 1. Lead Conversion — `models/lead_conversion.py`
- **What:** predicts the probability a lead will convert.
- **Algorithm:** `RandomForestClassifier` (200 trees).
- **Metrics:** accuracy, ROC-AUC, classification report.
- **Why RF:** handles mixed features well and gives us feature importances
  for the "why?" panel on the Predictions page.

### 2. Customer Churn — `models/churn_prediction.py`
- **What:** for converted customers, predicts the probability they churn.
- **Algorithm:** `RandomForestClassifier`.
- **Notes:** trained only on rows where `Converted == 1` so we don't
  bias the model with irrelevant non-customers.

### 3. Sales Forecast — `models/sales_forecast.py`
- **What:** monthly revenue forecast for the next N months.
- **Algorithm:** `LinearRegression` on `month_index → total_revenue`.
- **Notes:** intentionally simple so it's easy to explain and audit.

### 4. Customer Segmentation — `models/segmentation.py`
- **What:** groups customers into 4 clusters: Low-Value / Growing /
  Steady / High-Value.
- **Algorithm:** `KMeans` on scaled Budget / Engagement / Revenue.
- **Notes:** cluster IDs are re-ranked by centroid strength so labels
  are stable across re-trainings.

### 5. Lead Scoring — `models/lead_scoring.py`
- **What:** 0-100 score + Hot / Warm / Cold tier.
- **Approach:** transparent weighted formula (not ML) so salespeople can
  literally read the weight table and understand any score.

---

## Future improvements

- Swap the linear-regression forecast for Prophet or ARIMA when a longer
  history is available.
- Add automated tests (pytest) for the preprocessing pipeline and KPI
  calculations.
- Replace synthetic data with a live database connection (SQLAlchemy).
- Push feature importances through SHAP for per-lead explanations.
- Add a login layer via `streamlit-authenticator`.
- Deploy to Streamlit Community Cloud or a container platform.

---

## License

Provided as-is for educational and portfolio use.
