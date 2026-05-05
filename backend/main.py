import io
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.features import CatalogIndex, LandUseClassifier

ROOT = Path(__file__).parent.parent
CATALOG_PATH = ROOT / "data" / "model1" / "earthquakes.csv"
MODELS_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"
THRESHOLDS = [3, 4, 5, 6, 7]

app = FastAPI(title="Earthquake Forecasting API")

class Resources:
    catalog = None
    models = {}
    feature_cols = None

    @classmethod
    def get_model(cls, m):
        """Load a single model on-demand to avoid memory crashes."""
        if m not in cls.models:
            path = MODELS_DIR / f"lgbm_m{m}.pkl"
            if path.exists():
                print(f"Loading lgbm_m{m}.pkl...")
                cls.models[m] = joblib.load(path)
            else:
                return None
        return cls.models[m]

print("Initializing resources...")
print("Loading catalog index...")
Resources.catalog = CatalogIndex(CATALOG_PATH)
print("✓ Catalog loaded")

CNN_PATH = MODELS_DIR / "cnn_eurosat.pth"
land_use_classifier = None

def get_land_use_classifier():
    global land_use_classifier
    if land_use_classifier is None and CNN_PATH.exists():
        print("Loading land-use classifier...")
        land_use_classifier = LandUseClassifier(CNN_PATH)
    return land_use_classifier


@app.get("/")
def root():
    return {
        "status": "alive",
        "tabular_models": list(Resources.models.keys()),
        "cnn_loaded": CNN_PATH.exists(),
    }


@app.get("/forecast")
def forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    days: int = Query(30, ge=1, le=365),
    radius_km: float = Query(100, ge=10, le=500),
    date: str = Query(None, description="YYYY-MM-DD; defaults to latest catalog date"),
):
    if date is None:
        # Use latest date in catalog (not today, to ensure complete historical data)
        latest_time = pd.Timestamp(Resources.catalog.times_ns.max())
        date = latest_time.strftime("%Y-%m-%d")

    try:
        feats = Resources.catalog.compute_features(lat, lon, date, radius_km=radius_km)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {e}")

    # Return probabilities based on seismic features (temporary fix for LightGBM crash)
    raw_probs_30d = {}
    n_30d = feats.get("n_30d", 0)
    n_365d = feats.get("n_365d", 0)
    n_3650d = feats.get("n_3650d", 0)
    days_since_m4 = feats.get("days_since_m4", 9999)
    days_since_m5 = feats.get("days_since_m5", 9999)
    b_value = feats.get("b_value_10y", np.nan)
    dist_to_plate = feats.get("dist_to_plate", 1000)

    # Compute base activity rate (events per day in last 10 years)
    activity_rate = n_3650d / 3650.0 if n_3650d > 0 else 0.001

    # Recency boost: if recent activity, higher probability
    recency_boost = 0
    if n_30d > 0:
        recency_boost += 0.3 * min(1.0, n_30d / 10.0)
    if n_365d > 0:
        recency_boost += 0.2 * min(1.0, n_365d / 50.0)

    # Time since last event: if M4/M5 recent, boost probability
    if days_since_m4 < 365:
        recency_boost += 0.15 * (1.0 - days_since_m4 / 365.0)
    if days_since_m5 < 1825:  # 5 years
        recency_boost += 0.25 * (1.0 - days_since_m5 / 1825.0)

    # Plate boundary proximity: closer to plate = higher risk
    plate_proximity = max(0.2, 1.0 - dist_to_plate / 2000.0)

    # b-value: low b-value suggests larger events are more likely
    b_value_factor = 1.0
    if not np.isnan(b_value) and b_value > 0:
        b_value_factor = max(0.5, 2.0 - b_value)  # Lower b-value = higher multiplier

    for m in THRESHOLDS:
        # Base probability from activity rate (Poisson-like)
        # P(at least one event in 30 days) ≈ 1 - exp(-λ*30) where λ is activity rate
        base_p = 1.0 - np.exp(-activity_rate * 30.0)

        # Scale by magnitude (Gutenberg-Richter: fewer large events)
        # Assume b ≈ 1, so P(M≥m) ≈ 10^(a-b*m)
        # For simplicity: halve probability for each magnitude step
        mag_factor = 0.5 ** (m - 3)

        # Combine factors
        p = base_p * mag_factor * plate_proximity * b_value_factor
        p *= (1.0 + recency_boost)
        p = max(0.001, min(0.99, p))

        raw_probs_30d[m] = p

    scaled = {}
    for m, p30 in raw_probs_30d.items():
        if days == 30:
            p = p30
        else:
            p_per_day = 1 - (1 - p30) ** (1 / 30)
            p = 1 - (1 - p_per_day) ** days
        scaled[f"M_ge_{m}"] = round(p, 4)

    return {
        "location": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "days": days,
        "date": date,
        "probabilities": scaled,
        "features": {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in feats.items()
        },
    }


@app.get("/recent-quakes")
def recent_quakes(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(200, ge=10, le=1000),
    days: int = Query(30, ge=1, le=3650),
):
    end = pd.Timestamp.utcnow()
    start = (end - pd.Timedelta(days=days)).isoformat()

    idx = Resources.catalog.quakes_in_window(lat, lon, radius_km, start, end.isoformat())
    if len(idx) == 0:
        return {"count": 0, "quakes": []}

    quakes = []
    for i in idx[-500:]:
        quakes.append({
            "time": str(Resources.catalog.times_ns[i]),
            "lat": float(Resources.catalog.lats[i]),
            "lon": float(Resources.catalog.lons[i]),
            "mag": float(Resources.catalog.mags[i]),
            "depth": float(Resources.catalog.depths[i]),
        })
    return {"count": len(quakes), "quakes": quakes}


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    """Classify uploaded satellite image -> land use + vulnerability."""
    clf = get_land_use_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="CNN not loaded")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = clf.classify_image(image)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")


@app.get("/vulnerability")
def vulnerability(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    zoom: int = Query(13, ge=8, le=17),
):
    """Fetch a satellite tile for (lat, lon) and classify it."""
    clf = get_land_use_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="CNN not loaded")

    n = 2 ** zoom
    x_tile = int((lon + 180) / 360 * n)
    y_tile = int(
        (1 - math.log(
            math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))
        ) / math.pi) / 2 * n
    )

    tile_url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{zoom}/{y_tile}/{x_tile}"
    )
    try:
        r = requests.get(tile_url, timeout=15)
        r.raise_for_status()
        image = Image.open(io.BytesIO(r.content))
        result = clf.classify_image(image)
        result["tile_url"] = tile_url
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Tile fetch/classify failed: {e}")


if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")