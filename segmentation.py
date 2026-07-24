"""
Customer segmentation with K-Means clustering.

Given each customer's engagement + budget profile, we assign them to one
of K clusters. This is a classic unsupervised task and gives sales teams
an at-a-glance view of who their customers actually are.

We keep the number of features small (3) so segments stay interpretable
and clusters can be plotted directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import settings
from models._storage import load_bundle, save_bundle


MODEL_NAME = "segmentation"

# Features that describe how a customer behaves.
SEGMENTATION_FEATURES = ["Budget", "Total_Engagement", "Revenue"]

# Human-readable labels ordered by cluster centroid strength.
# (We rank clusters by the sum of their centroid coordinates.)
SEGMENT_LABELS = ["Low-Value", "Growing", "Steady", "High-Value"]


@dataclass
class SegmentationResult:
    model: KMeans
    scaler: StandardScaler
    n_clusters: int
    cluster_label_map: dict[int, str]  # cluster_id -> human label
    silhouette: float | None = None


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the numeric DataFrame we'll feed to K-Means."""
    working = df.copy()
    working["Total_Engagement"] = (
        working.get("Meetings", 0)
        + working.get("Emails", 0)
        + working.get("Website_Visits", 0)
    )
    return working[SEGMENTATION_FEATURES].fillna(0)


def train_segmentation_model(df: pd.DataFrame, n_clusters: int = 4) -> SegmentationResult:
    """Fit K-Means on customer-level engagement/budget features."""
    features = _prepare_features(df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # `n_init="auto"` uses the default of running multiple initialisations
    # and keeping the best — safer than a single random start.
    model = KMeans(
        n_clusters=n_clusters,
        random_state=settings.RANDOM_SEED,
        n_init="auto",
    )
    cluster_ids = model.fit_predict(scaled)

    # Rank clusters by centroid magnitude so "High-Value" always corresponds
    # to the strongest cluster regardless of K-Means' arbitrary numbering.
    centroid_scores = model.cluster_centers_.sum(axis=1)
    ranked = np.argsort(centroid_scores)  # ascending
    labels = SEGMENT_LABELS[:n_clusters]
    cluster_label_map = {int(cid): label for cid, label in zip(ranked, labels)}

    silhouette = None
    # Silhouette only makes sense with 2+ clusters and enough samples.
    if n_clusters >= 2 and len(features) >= 50:
        try:
            from sklearn.metrics import silhouette_score
            silhouette = float(silhouette_score(scaled, cluster_ids, sample_size=1000, random_state=settings.RANDOM_SEED))
        except Exception:
            # Silhouette is a nice-to-have, not a blocker.
            silhouette = None

    return SegmentationResult(
        model=model,
        scaler=scaler,
        n_clusters=n_clusters,
        cluster_label_map=cluster_label_map,
        silhouette=silhouette,
    )


def save_segmentation_model(result: SegmentationResult) -> None:
    save_bundle(
        MODEL_NAME,
        {
            "model": result.model,
            "scaler": result.scaler,
            "n_clusters": result.n_clusters,
            "cluster_label_map": result.cluster_label_map,
            "silhouette": result.silhouette,
        },
    )


def load_or_train(df: pd.DataFrame, n_clusters: int = 4) -> dict[str, Any]:
    bundle = load_bundle(MODEL_NAME)
    if bundle is not None:
        return bundle
    result = train_segmentation_model(df, n_clusters=n_clusters)
    save_segmentation_model(result)
    return load_bundle(MODEL_NAME)


def assign_segments(bundle: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """Assign a human-readable segment label to every row in `df`."""
    scaler: StandardScaler = bundle["scaler"]
    model: KMeans = bundle["model"]
    label_map: dict[int, str] = bundle["cluster_label_map"]

    features = _prepare_features(df)
    scaled = scaler.transform(features)
    ids = model.predict(scaled)
    return pd.Series([label_map.get(int(i), "Unknown") for i in ids], index=df.index, name="Segment")
