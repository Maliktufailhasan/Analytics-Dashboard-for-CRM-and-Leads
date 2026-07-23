"""
Reusable preprocessing pipeline for the CRM dataset.

The same preprocessing must happen at training time AND at prediction time.
If they diverge (e.g. training scales a feature but prediction doesn't),
the model will silently produce wrong answers.

To keep them in sync we expose one class, `CRMPreprocessor`, which:
  - is `fit()` on training data (learns encoders, scalers, feature stats)
  - is `transform()`ed on any future data (training set, test set, single row)

The class uses simple building blocks (LabelEncoder, StandardScaler) instead
of scikit-learn's ColumnTransformer + Pipeline so that a beginner can trace
what happens to each column line-by-line.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Columns that describe *who* the lead is (categorical, low-cardinality).
CATEGORICAL_COLUMNS = [
    "Industry",
    "Country",
    "Lead_Source",
    "Company_Size",
    "Salesperson",
    "Status",
]

# Columns that describe *how much* activity there was (numeric).
NUMERIC_COLUMNS = [
    "Budget",
    "Meetings",
    "Emails",
    "Website_Visits",
]

# Columns we never want to feed to a model (identifiers, raw dates, targets).
DROP_COLUMNS = [
    "Lead_ID",
    "Customer_Name",
    "Last_Contact",
    "Created_Date",
]


class CRMPreprocessor:
    """Fit-once / transform-many preprocessor for CRM rows.

    Typical usage:
        pre = CRMPreprocessor().fit(df_train)
        X_train = pre.transform(df_train)
        X_test  = pre.transform(df_test)
    """

    def __init__(self, scale_numeric: bool = True) -> None:
        self.scale_numeric = scale_numeric

        # Per-column encoders, populated during fit().
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler | None = None

        # Median values used to fill missing numeric cells.
        self.numeric_medians: dict[str, float] = {}

        # Final feature order — models expect columns in a stable order.
        self.feature_names_: list[str] = []
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "CRMPreprocessor":
        """Learn encoders, scalers, and imputation values from `df`."""
        df = df.copy()

        # 1. Deduplicate — a duplicate row would give the model a wrong prior.
        df = df.drop_duplicates()

        # 2. Numeric imputation: remember the median of each numeric column
        #    so we can fill NaNs at transform() time too.
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                self.numeric_medians[col] = float(df[col].median())

        # 3. Fit a LabelEncoder for every categorical column.
        #    We also include an "Unknown" sentinel so unseen future categories
        #    don't crash transform().
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue
            values = df[col].fillna("Unknown").astype(str).tolist()
            values.append("Unknown")  # guarantee the sentinel is known
            encoder = LabelEncoder()
            encoder.fit(values)
            self.label_encoders[col] = encoder

        # 4. Add engineered features BEFORE dropping raw dates so
        #    `Days_Since_Created` can use `Created_Date`.
        df = self._add_engineered_features(df)

        # 5. Drop identifier / raw-date columns — they're not model features.
        df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

        # 6. Fit the numeric scaler on all numeric + engineered columns.
        numeric_all = self._numeric_feature_columns(df)
        if self.scale_numeric and numeric_all:
            self.scaler = StandardScaler().fit(df[numeric_all].fillna(0))

        # 7. Record the final feature order.
        self.feature_names_ = self._final_feature_order(df)
        self.is_fitted_ = True
        return self

    # -------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all learned transformations to `df` and return a numeric DataFrame."""
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() before .transform().")

        df = df.copy()

        # Impute numeric NaNs with training-set medians.
        for col, median_value in self.numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_value)

        # Encode categoricals, mapping unseen labels to "Unknown".
        for col, encoder in self.label_encoders.items():
            if col not in df.columns:
                continue
            known = set(encoder.classes_)
            values = (
                df[col]
                .fillna("Unknown")
                .astype(str)
                .apply(lambda v: v if v in known else "Unknown")
            )
            df[col] = encoder.transform(values)

        # Add engineered features (this needs Created_Date, so do it BEFORE dropping).
        df = self._add_engineered_features(df)

        # Now drop identifier / raw-date columns so the model input matches fit().
        df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

        # Scale numeric columns using the fitted scaler.
        numeric_all = self._numeric_feature_columns(df)
        if self.scale_numeric and self.scaler is not None and numeric_all:
            df[numeric_all] = self.scaler.transform(df[numeric_all].fillna(0))

        # Return columns in the exact order the models were trained on.
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0  # missing engineered column at prediction time
        return df[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper: fit on `df` then return its transform."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------ internals
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a few derived columns the models can benefit from.

        These are simple ratios and totals — nothing exotic. We keep them
        obvious so a reader can see exactly what "engagement" means.
        """
        df = df.copy()

        if {"Meetings", "Emails", "Website_Visits"}.issubset(df.columns):
            df["Total_Engagement"] = (
                df["Meetings"] + df["Emails"] + df["Website_Visits"]
            )

        if {"Budget", "Meetings"}.issubset(df.columns):
            # Add 1 to the denominator so we never divide by zero.
            df["Budget_Per_Meeting"] = df["Budget"] / (df["Meetings"] + 1)

        # Days since the lead was created — a quick recency signal.
        if "Created_Date" in df.columns:
            created = pd.to_datetime(df["Created_Date"], errors="coerce")
            df["Days_Since_Created"] = (
                pd.Timestamp.today().normalize() - created
            ).dt.days.fillna(0)

        return df

    def _numeric_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """List of columns the scaler should touch — base numerics + engineered."""
        engineered = ["Total_Engagement", "Budget_Per_Meeting", "Days_Since_Created"]
        return [c for c in (NUMERIC_COLUMNS + engineered) if c in df.columns]

    def _final_feature_order(self, df: pd.DataFrame) -> list[str]:
        """Deterministic feature order used everywhere the model touches data."""
        # We deliberately exclude target columns so callers can't leak labels.
        target_like = {"Converted", "Churned", "Revenue"}
        return [c for c in df.columns if c not in target_like]


def split_features_and_target(
    df: pd.DataFrame, target: str, drop_extra: Iterable[str] = ()
) -> tuple[pd.DataFrame, pd.Series]:
    """Return `(X, y)` where `y` is `target` and `X` is everything else."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is missing from the DataFrame.")
    y = df[target]
    X = df.drop(columns=[target, *drop_extra], errors="ignore")
    return X, y
