"""
Train a Random Forest classifier for water pollution detection.
Uses real spectral index samples exported from GEE for known water bodies.
Falls back to synthetic data if GEE sampling fails.
Saves model to model.pkl for use in app.py.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

# ─── Known water body reference points ───────────────────────────────────────
# Format: (lat, lng, label)  0=Safe, 1=Moderate, 2=Polluted
REFERENCE_POINTS = [
    # Clean / Safe water bodies
    (46.8182,  8.2275,  0),   # Lake Lucerne, Switzerland
    (44.4268,  26.1025, 0),   # Lake Snagov, Romania
    (-3.3869,  36.6958, 0),   # Lake Manyara, Tanzania
    (61.9241,  25.7482, 0),   # Lake Paijanne, Finland
    (47.5596,  13.6493, 0),   # Wolfgangsee, Austria
    (46.4312,   6.5765, 0),   # Lake Geneva, Switzerland
    (-41.7774, 172.8344,0),   # Lake Rotoiti, New Zealand
    (58.3806,  26.7251, 0),   # Lake Vortsjarv, Estonia
    (64.5000, -21.0000, 0),   # Thingvallavatn, Iceland
    (45.9432,  24.9668, 0),   # Lake Balea, Romania

    # Moderate pollution
    (28.6139,  77.2090, 1),   # Yamuna River, Delhi
    (31.5497,  74.3436, 1),   # Ravi River, Lahore
    (23.7275,  90.4070, 1),   # Buriganga River, Dhaka
    (19.0760,  72.8777, 1),   # Thane Creek, Mumbai
    (30.0444,  31.2357, 1),   # Nile near Cairo
    (22.5726,  88.3639, 1),   # Hooghly River, Kolkata
    (13.0827,  80.2707, 1),   # Cooum River, Chennai
    (17.3850,  78.4867, 1),   # Musi River, Hyderabad
    (12.9716,  77.5946, 1),   # Bellandur Lake, Bangalore
    (18.5204,  73.8567, 1),   # Mula-Mutha River, Pune

    # Heavily polluted
    (25.5941,  85.1376, 2),   # Ganga near Patna
    (26.8467,  80.9462, 2),   # Gomti River, Lucknow
    (22.3072,  73.1812, 2),   # Vishwamitri River, Vadodara
    (23.2599,  77.4126, 2),   # Betwa River, Bhopal
    (21.1458,  79.0882, 2),   # Nag River, Nagpur
    (28.4595,  77.0266, 2),   # Najafgarh Lake drain, Delhi
    (10.8505,  76.2711, 2),   # Chaliyar River, Kerala
    (16.5062,  80.6480, 2),   # Krishna River industrial zone
    (20.9374,  85.0980, 2),   # Brahmani River, Odisha industrial
    (22.8046,  86.2029, 2),   # Subarnarekha River, Jharkhand
]


# ─── GEE Sampling ─────────────────────────────────────────────────────────────
def sample_gee_indices(points):
    """Sample NDWI, NDTI, FAI from GEE Sentinel-2 for each reference point."""
    import ee

    key_json   = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
    project_id = os.getenv("GEE_PROJECT_ID", "")

    if key_json:
        key_data = json.loads(key_json)
        credentials = ee.ServiceAccountCredentials(
            email=key_data["client_email"],
            key_data=json.dumps(key_data),
        )
        ee.Initialize(credentials=credentials, project=project_id)
    else:
        ee.Initialize(project=project_id)

    logger.info("GEE initialized, sampling %d reference points...", len(points))

    records = []
    end_date   = "2024-12-01"
    start_date = "2024-06-01"

    for lat, lng, label in points:
        try:
            aoi = ee.Geometry.Point([lng, lat]).buffer(3000)

            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            )

            count = collection.size().getInfo()
            if count == 0:
                logger.warning("No images for (%.4f, %.4f), skipping", lat, lng)
                continue

            composite = collection.median().clip(aoi)

            # Compute indices
            ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
            ndti = composite.normalizedDifference(["B4", "B3"]).rename("NDTI")
            nir  = composite.select("B8")
            red  = composite.select("B4")
            swir = composite.select("B11")
            fai  = nir.subtract(red.add(swir.subtract(red).multiply(
                (833 - 665) / (1614 - 665)
            ))).rename("FAI")

            stats = (
                ndwi.addBands(ndti).addBands(fai)
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=20,
                    maxPixels=1e8,
                )
                .getInfo()
            )

            if stats.get("NDWI") is None:
                logger.warning("No stats for (%.4f, %.4f), skipping", lat, lng)
                continue

            records.append({
                "ndwi":  round(stats["NDWI"], 4),
                "ndti":  round(stats["NDTI"], 4),
                "fai":   round(stats["FAI"],  6),
                "label": label,
            })
            logger.info("Sampled (%.4f, %.4f) → NDWI=%.3f NDTI=%.3f FAI=%.5f label=%d",
                        lat, lng, stats["NDWI"], stats["NDTI"], stats["FAI"], label)

        except Exception as e:
            logger.warning("Failed (%.4f, %.4f): %s", lat, lng, e)
            continue

    return pd.DataFrame(records)


# ─── Synthetic Fallback ───────────────────────────────────────────────────────
def generate_synthetic_data(n=600):
    """Generate realistic synthetic training data based on water index physics."""
    rng = np.random.default_rng(42)
    rows = []

    # Safe: high NDWI, low NDTI, low FAI
    for _ in range(n // 3):
        rows.append({
            "ndwi":  rng.uniform(0.3,  0.8),
            "ndti":  rng.uniform(-0.15, 0.0),
            "fai":   rng.uniform(-0.01, 0.005),
            "label": 0,
        })
    # Moderate
    for _ in range(n // 3):
        rows.append({
            "ndwi":  rng.uniform(0.1,  0.35),
            "ndti":  rng.uniform(0.0,  0.1),
            "fai":   rng.uniform(0.0,  0.02),
            "label": 1,
        })
    # Polluted
    for _ in range(n // 3):
        rows.append({
            "ndwi":  rng.uniform(-0.2, 0.15),
            "ndti":  rng.uniform(0.08, 0.35),
            "fai":   rng.uniform(0.015, 0.08),
            "label": 2,
        })

    return pd.DataFrame(rows)


# ─── Train & Save ─────────────────────────────────────────────────────────────
def train(df: pd.DataFrame, out_path="model.pkl"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder

    X = df[["ndwi", "ndti", "fai"]].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    logger.info("\n%s", classification_report(y_test, preds,
        target_names=["Safe", "Moderate", "Polluted"]))

    with open(out_path, "wb") as f:
        pickle.dump(clf, f)

    logger.info("Model saved to %s", out_path)
    return clf


if __name__ == "__main__":
    # 1. Try GEE real data
    df_gee = pd.DataFrame()
    try:
        df_gee = sample_gee_indices(REFERENCE_POINTS)
        logger.info("GEE samples collected: %d", len(df_gee))
    except Exception as e:
        logger.warning("GEE sampling failed: %s — using synthetic only", e)

    # 2. Always add synthetic data (augments real data or acts as sole source)
    df_synthetic = generate_synthetic_data(n=600)
    logger.info("Synthetic samples: %d", len(df_synthetic))

    # 3. Combine — real data gets 3x weight via repetition if available
    if len(df_gee) >= 10:
        df = pd.concat([df_gee] * 3 + [df_synthetic], ignore_index=True)
        logger.info("Training on %d samples (GEE x3 + synthetic)", len(df))
    else:
        df = df_synthetic
        logger.info("Training on synthetic data only (%d samples)", len(df))

    train(df)
