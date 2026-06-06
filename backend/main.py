import io
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
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

FEATURE_COLS = [
    "lat", "lon",
    "n_30d", "n_90d", "n_365d", "n_3650d",
    "max_mag_365d", "mean_mag_365d",
    "days_since_m4", "days_since_m5",
    "b_value_10y", "a_value_10y",
    "mean_depth_365d", "dist_to_plate",
    "n_m5_ring_10y", "n_m6_ring_10y",
    "dist_to_nearest_m5_10y", "dist_to_nearest_m6_ever",
]

app = FastAPI(title="Earthquake Forecasting API")

class Resources:
    catalog = None
    sessions = {}
    feature_cols = FEATURE_COLS

    @classmethod
    def get_session(cls, m):
        if m not in cls.sessions:
            path = MODELS_DIR / f"lgbm_m{m}.onnx"
            if path.exists():
                print(f"Loading lgbm_m{m}.onnx...")
                cls.sessions[m] = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
            else:
                return None
        return cls.sessions[m]

print("Initializing resources...")
print("Loading catalog index...")
Resources.catalog = CatalogIndex(CATALOG_PATH)
print("✓ Catalog loaded")

print("Loading LightGBM ONNX models...")
for m in THRESHOLDS:
    sess = Resources.get_session(m)
    if sess is not None:
        print(f"✓ lgbm_m{m}.onnx loaded")
    else:
        print(f"✗ lgbm_m{m}.onnx not found")

CNN_PATH = MODELS_DIR / "cnn_eurosat.xml"
land_use_classifier = None

def get_land_use_classifier():
    global land_use_classifier
    if land_use_classifier is None:
        try:
            print("Loading land-use classifier...")
            land_use_classifier = LandUseClassifier(MODELS_DIR / "cnn_eurosat.pth")
        except Exception as e:
            print(f"Failed to load CNN: {e}")
    return land_use_classifier


@app.get("/")
def root():
    return {
        "status": "alive",
        "tabular_models": list(Resources.sessions.keys()),
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
        latest_time = pd.Timestamp(Resources.catalog.times_ns.max())
        date = latest_time.strftime("%Y-%m-%d")

    try:
        feats = Resources.catalog.compute_features(lat, lon, date, radius_km=radius_km)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {e}")

    x = np.array(
        [[feats.get(col, np.nan) for col in Resources.feature_cols]],
        dtype=np.float32,
    )
    x = np.nan_to_num(x, nan=0.0, posinf=9999.0, neginf=-9999.0)

    raw_probs_30d = {}
    for m in THRESHOLDS:
        sess = Resources.get_session(m)
        if sess is None:
            continue
        try:
            outputs = sess.run(None, {"input": x})
            probs = outputs[1]
            p = float(probs[0, 1])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model M≥{m} prediction failed: {e}")
        raw_probs_30d[m] = p

    if not raw_probs_30d:
        raise HTTPException(status_code=503, detail="No LightGBM models loaded")

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


# Sampling pattern: center point + two concentric rings (1 + 6 + 8 = 15 tiles).
# Each sample carries a weight so the clicked point and nearby area count more
# than the far edge of the circle — otherwise a city gets diluted by its rural
# surroundings and the exposure reads too low.
CENTER_WEIGHT = 4.0
VULN_RINGS = ((0.5, 6, 2.0), (0.95, 8, 1.0))  # (radius fraction, #points, weight each)


def _tile_xy(lat: float, lon: float, zoom: int):
    """Web-Mercator tile indices for a lat/lon at a given zoom."""
    lat = max(-85.0, min(85.0, lat))
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _weighted_samples(lat: float, lon: float, radius_km: float):
    """Center + ring points as (lat, lon, weight) covering the selected circle."""
    samples = [(lat, lon, CENTER_WEIGHT)]
    cos_lat = max(0.01, math.cos(math.radians(lat)))
    for frac, count, weight in VULN_RINGS:
        r_km = radius_km * frac
        dlat = r_km / 111.0
        for i in range(count):
            ang = 2 * math.pi * i / count
            plat = max(-85.0, min(85.0, lat + dlat * math.sin(ang)))
            plon = ((lon + (r_km / (111.0 * cos_lat)) * math.cos(ang) + 180.0) % 360.0) - 180.0
            samples.append((plat, plon, weight))
    return samples


def _fetch_tile(zoom: int, x: int, y: int):
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)), url


@app.get("/vulnerability")
def vulnerability(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    # zoom 16 ⇒ tile footprint ~611 m, matching EuroSAT's ~640 m training scale.
    zoom: int = Query(16, ge=8, le=18),
    radius_km: float = Query(100, ge=10, le=500),
):
    clf = get_land_use_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="CNN not loaded")

    # Map weighted sample points to tiles, accumulating weight per unique tile
    # (tiles that coincide at small radii sum their weights).
    tile_weights = {}
    for plat, plon, w in _weighted_samples(lat, lon, radius_km):
        key = _tile_xy(plat, plon, zoom)
        tile_weights[key] = tile_weights.get(key, 0.0) + w

    def fetch(key):
        x, y = key
        try:
            return key, _fetch_tile(zoom, x, y)
        except Exception:
            return key, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        fetched = dict(ex.map(fetch, list(tile_weights.keys())))

    # Classify each fetched tile (sequentially — inference is fast, avoids
    # sharing one inference session across threads) and combine by weighted average.
    scores, num, wsum, class_acc = [], 0.0, 0.0, None
    for key, payload in fetched.items():
        if payload is None:
            continue
        image, _url = payload
        try:
            r = clf.classify_image(image)
        except Exception:
            continue
        w = tile_weights[key]
        s = r["vulnerability_score"]
        scores.append(s)
        num += s * w
        wsum += w
        if class_acc is None:
            class_acc = {c: 0.0 for c in r["all_classes"]}
        for c, p in r["all_classes"].items():
            class_acc[c] += p * w

    if not scores:
        raise HTTPException(status_code=502, detail="No tiles could be fetched/classified")

    avg_score = num / wsum
    avg_classes = {c: round(v / wsum, 3) for c, v in class_acc.items()}
    dominant = max(avg_classes, key=avg_classes.get)

    return {
        "vulnerability_score": round(avg_score, 1),
        "predicted_class": dominant,
        "confidence": avg_classes[dominant],
        "all_classes": avg_classes,
        "n_samples": len(scores),
        "n_tiles": len(tile_weights),
        "radius_km": radius_km,
        "score_min": round(min(scores), 1),
        "score_max": round(max(scores), 1),
    }


if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")