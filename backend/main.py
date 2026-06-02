import io
import math
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


@app.get("/vulnerability")
def vulnerability(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    zoom: int = Query(13, ge=8, le=17),
):
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