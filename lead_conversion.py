"""
Lead conversion prediction.

Given a lead's attributes (budget, industry, engagement, ...) we predict
the probability that they will eventually convert into a paying customer.

Model choice: RandomForestClassifier.
  - Handles mixed numeric/encoded categorical features well.
  - Gives us feature importances essentially for free, which powers the
    "why did the model predict this?" section on the Predictions page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from config import settings
from models._storage import load_bundle, save_bundle
from preprocessing.pipeline import CRMPreprocessor, split_features_and_target


MODEL_NAME = "lead_conversion"
TARGET = "Converted"


@dataclass
class ConversionResult:
    """Everything needed to describe a trained conversion model."""
    model: RandomForestClassifier
    preprocessor: CRMPreprocessor
    accuracy: float
    roc_auc: float
    report: str
    feature_importances: dict[str, float]


def train_lead_conversion_model(df: pd.DataFrame) -> ConversionResult:
    """Fit the Random Forest classifier on the full CRM dataset."""
    # We only need columns the preprocessor understands; the target stays
    # separate so we don't accidentally leak it into X.
    preprocessor = CRMPreprocessor()
    processed = preprocessor.fit_transform(df.drop(columns=["Churned", "Revenue"], errors="ignore"))
    X, y = split_features_and_target(processed.assign(**{TARGET: df[TARGET].values}), TARGET)

    # Stratify only if the smaller class has enough members.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.RANDOM_SEED, stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=settings.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluation metrics we'll surface in the UI.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = float(accuracy_score(y_test, predictions))
    roc = float(roc_auc_score(y_test, probabilities))
    report = classification_report(y_test, predictions, digits=3)

    importances = dict(zip(X.columns, model.feature_importances_))

    return ConversionResult(
        model=model,
        preprocessor=preprocessor,
        accuracy=accuracy,
        roc_auc=roc,
        report=report,
        feature_importances=importances,
    )


def save_conversion_model(result: ConversionResult) -> None:
    """Persist a trained model bundle."""
    save_bundle(
        MODEL_NAME,
        {
            "model": result.model,
            "preprocessor": result.preprocessor,
            "accuracy": result.accuracy,
            "roc_auc": result.roc_auc,
            "report": result.report,
            "feature_importances": result.feature_importances,
        },
    )


def load_or_train(df: pd.DataFrame) -> dict[str, Any]:
    """Load a saved bundle if available; otherwise train, save, then return it."""
    bundle = load_bundle(MODEL_NAME)
    if bundle is not None:
        return bundle

    result = train_lead_conversion_model(df)
    save_conversion_model(result)
    return load_bundle(MODEL_NAME)  # re-load so the caller gets a dict shape


def predict_conversion_probability(bundle: dict[str, Any], lead_row: pd.DataFrame) -> float:
    """Return the conversion probability (0..1) for a single lead."""
    preprocessor: CRMPreprocessor = bundle["preprocessor"]
    model: RandomForestClassifier = bundle["model"]
    X = preprocessor.transform(lead_row)
    return float(model.predict_proba(X)[0, 1])
