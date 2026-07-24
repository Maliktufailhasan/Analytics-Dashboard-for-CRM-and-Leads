"""
Customer churn prediction.

For customers we already converted, we predict the probability that they
will churn (i.e. stop being a customer).

Model choice: RandomForestClassifier.
  - The task requirements allow XGBoost too, but sticking with scikit-learn
    keeps the dependency list short and avoids GPU/build surprises for
    people who just want to run the app.
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


MODEL_NAME = "churn_prediction"
TARGET = "Churned"


@dataclass
class ChurnResult:
    model: RandomForestClassifier
    preprocessor: CRMPreprocessor
    accuracy: float
    roc_auc: float
    report: str
    feature_importances: dict[str, float]


def train_churn_model(df: pd.DataFrame) -> ChurnResult:
    """Fit the churn model on the subset of rows that actually converted."""
    # Churn is only meaningful for converted customers — the rest are
    # non-events and would bias the model.
    customers = df[df["Converted"] == 1].copy()
    if customers.empty:
        raise ValueError("No converted customers available to train a churn model.")

    preprocessor = CRMPreprocessor()
    processed = preprocessor.fit_transform(
        customers.drop(columns=["Converted", "Revenue"], errors="ignore")
    )
    X, y = split_features_and_target(
        processed.assign(**{TARGET: customers[TARGET].values}), TARGET
    )

    # Stratify only when every class has at least 2 members — otherwise
    # scikit-learn raises. This matters on tiny/imbalanced samples.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.RANDOM_SEED, stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=settings.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = float(accuracy_score(y_test, predictions))
    roc = float(roc_auc_score(y_test, probabilities))
    report = classification_report(y_test, predictions, digits=3)
    importances = dict(zip(X.columns, model.feature_importances_))

    return ChurnResult(
        model=model,
        preprocessor=preprocessor,
        accuracy=accuracy,
        roc_auc=roc,
        report=report,
        feature_importances=importances,
    )


def save_churn_model(result: ChurnResult) -> None:
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
    bundle = load_bundle(MODEL_NAME)
    if bundle is not None:
        return bundle
    result = train_churn_model(df)
    save_churn_model(result)
    return load_bundle(MODEL_NAME)


def predict_churn_probability(bundle: dict[str, Any], lead_row: pd.DataFrame) -> float:
    """Return churn probability for a single (already-converted) customer."""
    preprocessor: CRMPreprocessor = bundle["preprocessor"]
    model: RandomForestClassifier = bundle["model"]
    X = preprocessor.transform(lead_row)
    return float(model.predict_proba(X)[0, 1])
