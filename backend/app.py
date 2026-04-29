"""
AquaWatch Backend - FastAPI application for satellite water pollution detection.
Deployed on Render. Uses Google Earth Engine Python API for Sentinel-2 analysis.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import ee
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aquawatch")

# ─── GEE Authentication ─────────────────────────────────────────────────────
def initialize_gee():
    """
    Authenticate with Google Earth Engine using a service account JSON key
    stored in the GEE_SERVICE_ACCOUNT_KEY environment variable (JSON string).
    Falls back to application-default credentials for local dev.
    """
    key_json = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
    project_id = os.getenv("GEE_PROJECT_ID", "")

    if key_json:
        try:
            key_data = json.loads(key_json)
            credentials = ee.ServiceAccountCredentials(
                email=key_data["client_email"],
                key_data=json.dumps(key_data),
            )
            ee.Initialize(credentials=credentials, project=project_id)
            logger.info("GEE initialized via service account.")
        except Exception as exc:
            logger.error("GEE service account init failed: %s", exc)
            raise RuntimeError(f"GEE init failed: {exc}") from exc
    else:
        # Local development: use `earthengine authenticate` credentials
        try:
            ee.Initialize(project=project_id)
            logger.info("GEE initialized via application-default credentials.")
        except Exception as exc:
            logger.error("GEE default init failed: %s", exc)
            raise RuntimeError(f"GEE init failed: {exc}") from exc


initialize_gee()

# ─── ML Model ────────────────────────────────────────────────────────────────
_MODEL = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

def load_model():
    global _MODEL
    if os.path.exists(_MODEL_PATH):
        with open(_MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)
        logger.info("ML model loaded from %s", _MODEL_PATH)
    else:
        logger.warning("model.pkl not found — using rule-based fallback. Run train_model.py to generate it.")

load_model()

# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AquaWatch API",
    description="Water pollution detection using Sentinel-2 satellite imagery via Google Earth Engine.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ───────────────────────────────────────────────────────────────
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_COVER_THRESHOLD = 20          # Max cloud cover % for image selection
DEFAULT_BUFFER_METERS = 5000        # AOI buffer radius in metres
NDWI_WATER_THRESHOLD = 0.0          # NDWI > 0 → water pixel
LOW_CLARITY_NDWI_THRESHOLD = 0.1
POLLUTION_NDTI_THRESHOLD = 0.1
ALGAE_FAI_THRESHOLD = 0.02
AREA_SCALE_METERS = 20
DEFAULT_BASELINE_MONTHS = 12
POLLUTION_THRESHOLDS = {
    "safe":      {"ndwi_min": 0.3,  "turbidity_max": 10},
    "moderate":  {"ndwi_min": 0.1,  "turbidity_max": 30},
    "polluted":  {"ndwi_min": -0.1, "turbidity_max": 100},
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def build_aoi(lat: float, lng: float, buffer_m: int = DEFAULT_BUFFER_METERS) -> ee.Geometry:
    """Return a circular AOI geometry around the given coordinates."""
    return ee.Geometry.Point([lng, lat]).buffer(buffer_m)


def mask_clouds_s2(image: ee.Image) -> ee.Image:
    """Mask clouds and cirrus in Sentinel-2 SR using the QA60 band."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000)


def compute_ndwi(image: ee.Image) -> ee.Image:
    """NDWI = (Green - NIR) / (Green + NIR)  — McFeeters 1996."""
    return image.normalizedDifference(["B3", "B8"]).rename("NDWI")


def compute_ndti(image: ee.Image) -> ee.Image:
    """
    NDTI (Normalised Difference Turbidity Index) = (Red - Green) / (Red + Green).
    Higher values indicate more turbid / potentially polluted water.
    """
    return image.normalizedDifference(["B4", "B3"]).rename("NDTI")


def compute_fai(image: ee.Image) -> ee.Image:
    """
    Floating Algae Index (FAI) proxy using NIR, Red, SWIR1.
    Positive FAI → algal bloom / surface scum.
    """
    nir   = image.select("B8")
    red   = image.select("B4")
    swir1 = image.select("B11")
    # Linear baseline between Red and SWIR1 at NIR wavelength
    fai = nir.subtract(
        red.add(swir1.subtract(red).multiply((832.8 - 664.6) / (1613.7 - 664.6)))
    ).rename("FAI")
    return fai


def classify_pollution(ndwi_val: float, ndti_val: float, fai_val: float) -> dict:
    """
    ML-based pollution classification using Random Forest.
    Falls back to rule-based thresholds if model is not loaded.
    """
    LABELS = ["Safe", "Moderate", "Polluted"]
    COLORS = {"Safe": "#27ae60", "Moderate": "#f39c12", "Polluted": "#e74c3c"}

    if _MODEL is not None:
        features = np.array([[ndwi_val, ndti_val, fai_val]])
        pred = int(_MODEL.predict(features)[0])
        proba = _MODEL.predict_proba(features)[0]
        label = LABELS[pred]
        score = int(round((1 - proba[0]) * 100))  # higher = more polluted

        # Derive factors from feature importances + values
        factors = []
        if ndwi_val < 0.1:
            factors.append("Low water clarity (NDWI)")
        elif ndwi_val < 0.3:
            factors.append("Moderate water clarity (NDWI)")
        if ndti_val > 0.1:
            factors.append("High turbidity (NDTI)")
        elif ndti_val > 0.0:
            factors.append("Moderate turbidity (NDTI)")
        if fai_val > 0.02:
            factors.append("Algal bloom detected (FAI)")
        elif fai_val > 0.005:
            factors.append("Possible algal activity (FAI)")

        return {"label": label, "score": min(score, 100), "color": COLORS[label], "factors": factors}

    # ── Rule-based fallback ──────────────────────────────────────────────────
    score = 0
    factors = []
    if ndwi_val < 0.1:
        score += 40; factors.append("Low water clarity (NDWI)")
    elif ndwi_val < 0.3:
        score += 20; factors.append("Moderate water clarity (NDWI)")
    if ndti_val > 0.1:
        score += 35; factors.append("High turbidity (NDTI)")
    elif ndti_val > 0.0:
        score += 15; factors.append("Moderate turbidity (NDTI)")
    if fai_val > 0.02:
        score += 25; factors.append("Algal bloom detected (FAI)")
    elif fai_val > 0.005:
        score += 10; factors.append("Possible algal activity (FAI)")

    label = "Polluted" if score >= 50 else "Moderate" if score >= 20 else "Safe"
    return {"label": label, "score": min(score, 100), "color": COLORS[label], "factors": factors}


def get_date_range(days_back: int = 60):
    """Return ISO date strings for a rolling window ending today."""
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round_optional(value, digits: int):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def pct_change(current: float, baseline: Optional[float]) -> Optional[float]:
    """Percent change with protection around near-zero baselines."""
    if baseline is None or abs(baseline) < 0.01:
        return None
    return ((current - baseline) / abs(baseline)) * 100


def build_quality_masks(ndwi_img: ee.Image, ndti_img: ee.Image, fai_img: ee.Image):
    """Return water and pollution-risk masks for area calculations and overlays."""
    water_mask = ndwi_img.gt(NDWI_WATER_THRESHOLD)
    pollution_mask = water_mask.And(
        ndti_img.gt(POLLUTION_NDTI_THRESHOLD)
        .Or(fai_img.gt(ALGAE_FAI_THRESHOLD))
        .Or(ndwi_img.lt(LOW_CLARITY_NDWI_THRESHOLD))
    )
    return water_mask.rename("water_mask"), pollution_mask.rename("pollution_mask")


def compute_area_impact(
    aoi: ee.Geometry,
    ndwi_img: ee.Image,
    ndti_img: ee.Image,
    fai_img: ee.Image,
) -> dict:
    """Estimate water area and potentially affected water area in the AOI."""
    water_mask, pollution_mask = build_quality_masks(ndwi_img, ndti_img, fai_img)
    pixel_area = ee.Image.pixelArea()
    one = ee.Image.constant(1)

    stats = (
        pixel_area.rename("aoi_area_m2")
        .addBands(pixel_area.updateMask(water_mask).rename("water_area_m2"))
        .addBands(pixel_area.updateMask(pollution_mask).rename("affected_area_m2"))
        .addBands(one.updateMask(water_mask).rename("water_pixel_count"))
        .addBands(one.updateMask(pollution_mask).rename("affected_pixel_count"))
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=AREA_SCALE_METERS,
            maxPixels=1e9,
        )
        .getInfo()
    )

    aoi_area_m2 = float(stats.get("aoi_area_m2") or 0)
    water_area_m2 = float(stats.get("water_area_m2") or 0)
    affected_area_m2 = float(stats.get("affected_area_m2") or 0)
    water_pixels = int(round(stats.get("water_pixel_count") or 0))
    affected_pixels = int(round(stats.get("affected_pixel_count") or 0))

    water_coverage_pct = (water_area_m2 / aoi_area_m2 * 100) if aoi_area_m2 else 0
    affected_water_pct = (affected_area_m2 / water_area_m2 * 100) if water_area_m2 else 0

    return {
        "aoi_area_ha": round(aoi_area_m2 / 10000, 2),
        "water_area_ha": round(water_area_m2 / 10000, 2),
        "affected_area_ha": round(affected_area_m2 / 10000, 2),
        "water_coverage_pct": round(water_coverage_pct, 1),
        "affected_water_pct": round(affected_water_pct, 1),
        "water_pixel_count": water_pixels,
        "affected_pixel_count": affected_pixels,
        "scale_m": AREA_SCALE_METERS,
    }


def compute_index_stats(
    aoi: ee.Geometry,
    ndwi_img: ee.Image,
    ndti_img: ee.Image,
    fai_img: ee.Image,
) -> dict:
    """Mean spectral indices over detected water pixels, not surrounding land."""
    water_mask = ndwi_img.gt(NDWI_WATER_THRESHOLD)
    return (
        ndwi_img.updateMask(water_mask)
        .addBands(ndti_img.updateMask(water_mask))
        .addBands(fai_img.updateMask(water_mask))
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=AREA_SCALE_METERS,
            maxPixels=1e9,
        )
        .getInfo()
    )


def get_baseline_context(
    aoi: ee.Geometry,
    current_start_date: str,
    baseline_months: int,
) -> dict:
    """
    Build a historical baseline ending before the current analysis window.
    This powers anomaly detection and before/after evidence layers.
    """
    baseline_months = int(clamp(baseline_months, 3, 36))
    baseline_end = datetime.strptime(current_start_date, "%Y-%m-%d")
    baseline_start = baseline_end - timedelta(days=baseline_months * 30)
    start_str = baseline_start.strftime("%Y-%m-%d")
    end_str = baseline_end.strftime("%Y-%m-%d")

    raw_collection = (
        ee.ImageCollection(SENTINEL2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_str, end_str)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_THRESHOLD))
    )

    count = int(raw_collection.size().getInfo())
    if count == 0:
        return {
            "available": False,
            "images": 0,
            "period": {"start": start_str, "end": end_str},
            "indices": None,
            "cloud_pct": None,
        }

    cloud_pct = raw_collection.aggregate_mean("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    composite = raw_collection.map(mask_clouds_s2).median().clip(aoi)
    ndwi_img = compute_ndwi(composite)
    ndti_img = compute_ndti(composite)
    fai_img = compute_fai(composite)
    stats = compute_index_stats(aoi, ndwi_img, ndti_img, fai_img)

    return {
        "available": True,
        "images": count,
        "period": {"start": start_str, "end": end_str},
        "cloud_pct": round_optional(cloud_pct, 1),
        "image": composite,
        "ndwi_img": ndwi_img,
        "ndti_img": ndti_img,
        "fai_img": fai_img,
        "indices": {
            "ndwi": round_optional(stats.get("NDWI"), 4),
            "ndti": round_optional(stats.get("NDTI"), 4),
            "fai": round_optional(stats.get("FAI"), 6),
        },
    }


def compute_anomaly(current_indices: dict, baseline: dict) -> dict:
    """Compare current optical water-quality indicators with historical baseline."""
    if not baseline.get("available") or not baseline.get("indices"):
        return {
            "status": "insufficient_baseline",
            "label": "No baseline",
            "score": 0,
            "signals": ["Not enough historical cloud-free imagery for baseline comparison."],
            "baseline_period": baseline.get("period"),
            "baseline_images": baseline.get("images", 0),
            "baseline_indices": None,
            "deltas": None,
        }

    base = baseline["indices"]
    ndwi_delta = current_indices["ndwi"] - (base.get("ndwi") or 0)
    ndti_delta = current_indices["ndti"] - (base.get("ndti") or 0)
    fai_delta = current_indices["fai"] - (base.get("fai") or 0)

    score = 0
    score += clamp(ndti_delta / 0.12, 0, 1) * 45
    score += clamp(fai_delta / 0.03, 0, 1) * 30
    score += clamp((-ndwi_delta) / 0.18, 0, 1) * 25
    score = int(round(clamp(score, 0, 100)))

    if score >= 60:
        status, label = "high_anomaly", "High anomaly"
    elif score >= 30:
        status, label = "watch", "Watch"
    else:
        status, label = "normal", "Normal"

    signals = []
    ndti_pct = pct_change(current_indices["ndti"], base.get("ndti"))
    fai_pct = pct_change(current_indices["fai"], base.get("fai"))
    ndwi_pct = pct_change(current_indices["ndwi"], base.get("ndwi"))

    if ndti_delta > 0.04:
        if ndti_pct is not None:
            signals.append(f"Turbidity signal is {ndti_pct:+.0f}% versus baseline.")
        else:
            signals.append(f"Turbidity signal increased by {ndti_delta:+.3f} NDTI.")
    if fai_delta > 0.008:
        if fai_pct is not None:
            signals.append(f"Algal/surface scum signal is {fai_pct:+.0f}% versus baseline.")
        else:
            signals.append(f"Algal/surface scum signal increased by {fai_delta:+.4f} FAI.")
    if ndwi_delta < -0.05:
        if ndwi_pct is not None:
            signals.append(f"Water clarity proxy dropped {abs(ndwi_pct):.0f}% versus baseline.")
        else:
            signals.append(f"Water clarity proxy dropped by {ndwi_delta:.3f} NDWI.")
    if not signals:
        signals.append("Current optical water-quality indicators are close to baseline.")

    return {
        "status": status,
        "label": label,
        "score": score,
        "signals": signals,
        "baseline_period": baseline.get("period"),
        "baseline_images": baseline.get("images", 0),
        "baseline_indices": base,
        "deltas": {
            "ndwi": round(ndwi_delta, 4),
            "ndti": round(ndti_delta, 4),
            "fai": round(fai_delta, 6),
            "ndwi_pct": round_optional(ndwi_pct, 1),
            "ndti_pct": round_optional(ndti_pct, 1),
            "fai_pct": round_optional(fai_pct, 1),
        },
    }


def compute_confidence(
    images_used: int,
    baseline_images: int,
    cloud_pct: Optional[float],
    impact: dict,
) -> dict:
    """Data-quality confidence score for the current risk assessment."""
    score = 35
    drivers = []

    score += min(images_used, 10) * 3.0
    drivers.append(f"{images_used} current Sentinel-2 scenes")

    if baseline_images:
        score += min(baseline_images, 30) * 0.7
        drivers.append(f"{baseline_images} baseline scenes")
    else:
        score -= 8
        drivers.append("limited baseline history")

    water_coverage = impact.get("water_coverage_pct", 0)
    if water_coverage >= 20:
        score += 15
        drivers.append("strong water-pixel coverage")
    elif water_coverage >= 5:
        score += 8
        drivers.append("moderate water-pixel coverage")
    else:
        score -= 10
        drivers.append("small visible water area")

    if cloud_pct is not None:
        score += max(0, CLOUD_COVER_THRESHOLD - cloud_pct) * 0.5
        drivers.append(f"{round(cloud_pct, 1)}% mean cloud cover")

    score = int(round(clamp(score, 25, 95)))
    level = "High" if score >= 75 else "Medium" if score >= 55 else "Low"
    return {"score": score, "level": level, "drivers": drivers}


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "AquaWatch API", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/analyze")
def analyze(
    lat: float = Query(..., description="Latitude of the point of interest"),
    lng: float = Query(..., description="Longitude of the point of interest"),
    buffer: int = Query(DEFAULT_BUFFER_METERS, description="AOI buffer radius in metres"),
    days_back: int = Query(60, description="Days of imagery to look back"),
    baseline_months: int = Query(DEFAULT_BASELINE_MONTHS, description="Historical baseline window in months"),
):
    """
    Analyse water quality at a given location using the most recent
    cloud-free Sentinel-2 composite within the specified window.

    Returns:
    - Pollution classification (Safe / Moderate / Polluted)
    - Historical anomaly score against a local baseline
    - Estimated water and affected surface area
    - Confidence score for data quality
    - NDWI, NDTI, FAI mean values
    - Tile URL for map overlay
    - Bounding box of the AOI
    """
    try:
        aoi = build_aoi(lat, lng, buffer)
        start_date, end_date = get_date_range(days_back)

        raw_collection = (
            ee.ImageCollection(SENTINEL2_COLLECTION)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_THRESHOLD))
        )
        collection = (
            raw_collection
            .map(mask_clouds_s2)
        )

        count = int(raw_collection.size().getInfo())
        if count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No cloud-free Sentinel-2 images found for this location in the last {days_back} days. "
                       "Try increasing days_back or choosing a different location.",
            )
        current_cloud_pct = raw_collection.aggregate_mean("CLOUDY_PIXEL_PERCENTAGE").getInfo()

        # Use median composite for robustness
        composite = collection.median().clip(aoi)

        ndwi_img = compute_ndwi(composite)
        ndti_img = compute_ndti(composite)
        fai_img  = compute_fai(composite)

        # Reduce to mean values over AOI
        stats = compute_index_stats(aoi, ndwi_img, ndti_img, fai_img)

        ndwi_val = stats.get("NDWI", 0) or 0
        ndti_val = stats.get("NDTI", 0) or 0
        fai_val  = stats.get("FAI",  0) or 0
        current_indices = {
            "ndwi": round(float(ndwi_val), 4),
            "ndti": round(float(ndti_val), 4),
            "fai":  round(float(fai_val),  6),
        }

        classification = classify_pollution(ndwi_val, ndti_val, fai_val)
        impact = compute_area_impact(aoi, ndwi_img, ndti_img, fai_img)
        baseline = get_baseline_context(aoi, start_date, baseline_months)
        anomaly = compute_anomaly(current_indices, baseline)
        confidence = compute_confidence(
            images_used=count,
            baseline_images=baseline.get("images", 0),
            cloud_pct=round_optional(current_cloud_pct, 1),
            impact=impact,
        )

        if anomaly["status"] in {"watch", "high_anomaly"}:
            classification["factors"].append(f"Historical anomaly: {anomaly['label']}")
        if impact["affected_area_ha"] > 0:
            classification["factors"].append(
                f"{impact['affected_area_ha']} ha of water pixels flagged"
            )

        # ── Tile URLs for map overlay ──────────────────────────────────────
        # True-colour RGB
        rgb_params = {
            "bands": ["B4", "B3", "B2"],
            "min": 0.0,
            "max": 0.3,
            "gamma": 1.4,
        }
        rgb_map = composite.getMapId(rgb_params)

        baseline_rgb_url = None
        if baseline.get("available"):
            baseline_rgb_url = baseline["image"].getMapId(rgb_params)["tile_fetcher"].url_format

        # NDWI coloured layer
        ndwi_params = {
            "bands": ["NDWI"],
            "min": -0.5,
            "max": 0.8,
            "palette": ["#8B4513", "#F5DEB3", "#87CEEB", "#1E90FF", "#00008B"],
        }
        ndwi_map = ndwi_img.getMapId(ndwi_params)

        water_mask, pollution_mask = build_quality_masks(ndwi_img, ndti_img, fai_img)

        # Pollution overlay (NDTI-based, masked to water pixels)
        pollution_params = {
            "bands": ["NDTI"],
            "min": -0.2,
            "max": 0.3,
            "palette": ["#27ae60", "#f39c12", "#e74c3c"],
        }
        pollution_map = ndti_img.updateMask(water_mask).getMapId(pollution_params)

        change_url = None
        if baseline.get("available"):
            change_img = (
                ndti_img.subtract(baseline["ndti_img"])
                .rename("NDTI_CHANGE")
                .updateMask(water_mask)
            )
            change_params = {
                "bands": ["NDTI_CHANGE"],
                "min": -0.08,
                "max": 0.15,
                "palette": ["#27ae60", "#f8fafc", "#f39c12", "#e74c3c"],
            }
            change_url = change_img.getMapId(change_params)["tile_fetcher"].url_format

        # AOI bounding box for map centering
        bounds = aoi.bounds().getInfo()["coordinates"][0]
        bbox = {
            "west":  bounds[0][0],
            "south": bounds[0][1],
            "east":  bounds[2][0],
            "north": bounds[2][1],
        }

        return {
            "location": {"lat": lat, "lng": lng},
            "aoi_buffer_m": buffer,
            "date_range": {"start": start_date, "end": end_date},
            "images_used": count,
            "mean_cloud_pct": round_optional(current_cloud_pct, 1),
            "indices": current_indices,
            "classification": classification,
            "anomaly": anomaly,
            "impact": impact,
            "confidence": confidence,
            "tile_urls": {
                "rgb":       rgb_map["tile_fetcher"].url_format,
                "baseline_rgb": baseline_rgb_url,
                "ndwi":      ndwi_map["tile_fetcher"].url_format,
                "pollution": pollution_map["tile_fetcher"].url_format,
                "change":    change_url,
            },
            "bbox": bbox,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /analyze: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/timeseries")
def timeseries(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    buffer: int = Query(DEFAULT_BUFFER_METERS, description="AOI buffer radius in metres"),
    months: int = Query(12, description="Number of months of history to fetch"),
):
    """
    Return monthly NDWI, NDTI, and FAI time-series for the given location.
    Each data point is the median composite for that calendar month.
    """
    try:
        aoi = build_aoi(lat, lng, buffer)
        end_date   = datetime.utcnow()
        start_date = end_date - timedelta(days=months * 30)

        collection = (
            ee.ImageCollection(SENTINEL2_COLLECTION)
            .filterBounds(aoi)
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_THRESHOLD))
            .map(mask_clouds_s2)
        )

        count = collection.size().getInfo()
        if count == 0:
            raise HTTPException(
                status_code=404,
                detail="No cloud-free images found for this location and time range.",
            )

        # Build monthly composites
        results = []
        current = start_date.replace(day=1)

        while current <= end_date:
            next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            month_str  = current.strftime("%Y-%m")

            monthly = (
                collection
                .filterDate(current.strftime("%Y-%m-%d"), next_month.strftime("%Y-%m-%d"))
                .median()
                .clip(aoi)
            )

            # Check if any images exist for this month
            month_count = (
                collection
                .filterDate(current.strftime("%Y-%m-%d"), next_month.strftime("%Y-%m-%d"))
                .size()
                .getInfo()
            )

            if month_count > 0:
                ndwi_img = compute_ndwi(monthly)
                ndti_img = compute_ndti(monthly)
                fai_img  = compute_fai(monthly)

                stats = (
                    ndwi_img.addBands(ndti_img).addBands(fai_img)
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=aoi,
                        scale=20,
                        maxPixels=1e9,
                    )
                    .getInfo()
                )

                ndwi_val = stats.get("NDWI", None)
                ndti_val = stats.get("NDTI", None)
                fai_val  = stats.get("FAI",  None)

                if ndwi_val is not None:
                    classification = classify_pollution(
                        ndwi_val or 0, ndti_val or 0, fai_val or 0
                    )
                    results.append({
                        "month":          month_str,
                        "ndwi":           round(ndwi_val, 4) if ndwi_val is not None else None,
                        "ndti":           round(ndti_val, 4) if ndti_val is not None else None,
                        "fai":            round(fai_val,  6) if fai_val  is not None else None,
                        "classification": classification["label"],
                        "score":          classification["score"],
                        "images":         month_count,
                    })

            current = next_month

        if not results:
            raise HTTPException(
                status_code=404,
                detail="Could not compute time-series. No valid water pixels found.",
            )

        # Trend: simple linear regression on NDWI
        ndwi_values = [r["ndwi"] for r in results if r["ndwi"] is not None]
        trend = "stable"
        if len(ndwi_values) >= 3:
            x = np.arange(len(ndwi_values), dtype=float)
            y = np.array(ndwi_values, dtype=float)
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.005:
                trend = "improving"
            elif slope < -0.005:
                trend = "degrading"

        return {
            "location":   {"lat": lat, "lng": lng},
            "months":     months,
            "data_points": len(results),
            "trend":      trend,
            "series":     results,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /timeseries: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/alerts")
def alerts(
    lat: float = Query(..., description="Latitude"),
    lng: float = Query(..., description="Longitude"),
    buffer: int = Query(DEFAULT_BUFFER_METERS, description="AOI buffer radius in metres"),
    baseline_months: int = Query(DEFAULT_BASELINE_MONTHS, description="Historical baseline window in months"),
):
    """
    Check current pollution status and return alert level + recommended actions.
    """
    try:
        result = analyze(lat=lat, lng=lng, buffer=buffer, days_back=60, baseline_months=baseline_months)
        classification = result["classification"]
        indices        = result["indices"]
        anomaly        = result.get("anomaly", {})
        impact         = result.get("impact", {})
        confidence     = result.get("confidence", {})

        alert_level = classification["label"]
        recommendations = []

        if alert_level == "Polluted":
            recommendations = [
                "⚠️ Avoid recreational water contact immediately.",
                "🚰 Do not use this water source for drinking or irrigation.",
                "📢 Notify local environmental authorities.",
                "🔬 Collect water samples for laboratory analysis.",
                "📍 Mark area as restricted until further assessment.",
            ]
        elif alert_level == "Moderate":
            recommendations = [
                "⚠️ Exercise caution near this water body.",
                "🔍 Monitor water quality over the next 2–4 weeks.",
                "📊 Increase sampling frequency.",
                "🏊 Limit recreational activities.",
            ]
        else:
            recommendations = [
                "✅ Water quality appears normal.",
                "📅 Continue routine monitoring.",
                "📈 Track seasonal variations.",
            ]

        if anomaly.get("status") == "high_anomaly":
            recommendations.insert(0, "Investigate sudden deviation from historical baseline.")
        elif anomaly.get("status") == "watch":
            recommendations.insert(0, "Schedule follow-up monitoring for baseline deviation.")

        if impact.get("affected_area_ha", 0) >= 1:
            recommendations.append(
                f"Prioritise field sampling across the flagged {impact['affected_area_ha']} ha area."
            )

        if confidence.get("level") == "Low":
            recommendations.append("Repeat analysis with clearer imagery before public-health escalation.")

        return {
            "location":        {"lat": lat, "lng": lng},
            "alert_level":     alert_level,
            "alert_color":     classification["color"],
            "pollution_score": classification["score"],
            "factors":         classification["factors"],
            "indices":         indices,
            "anomaly":         anomaly,
            "impact":          impact,
            "confidence":      confidence,
            "recommendations": recommendations,
            "timestamp":       datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /alerts: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
