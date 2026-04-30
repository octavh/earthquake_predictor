from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.features import CatalogIndex

ROOT = Path(__file__).parent.parent
CATALOG_PATH = ROOT / "data" / "earthquakes.csv"
MODELS_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"
THRESHOLDS = [3, 4, 5, 6, 7]

app = FastAPI(title="Earthquake Forecasting API")

print("Loading catalog index...")
catalog = CatalogIndex(CATALOG_PATH)

print("Loading trained models...")
models = {}
for m in THRESHOLDS:
    path = MODELS_DIR / f"lgbm_m{m}.pkl"
    if path.exists():
        models[m] = joblib.load(path)
        print(f"  Loaded lgbm_m{m}.pkl")
    else:
        print(f"  Missing {path.name} — skipping")

if not models:
    raise RuntimeError("No models found. Run train_tabular.py first.")

FEATURE_COLS = next(iter(models.values()))["feature_cols"]


@app.get("/")
def root():
    return {"status": "alive", "models_loaded": list(models.keys())}


@app.get("/forecast")
def forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    days: int = Query(30, ge=1, le=365),
    radius_km: float = Query(100, ge=10, le=500),
    date: str = Query(None, description="YYYY-MM-DD; defaults to today"),
):
    if date is None:
        date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    try:
        feats = catalog.compute_features(lat, lon, date, radius_km=radius_km)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {e}")

    X = pd.DataFrame([{c: feats[c] for c in FEATURE_COLS}])

    raw_probs_30d = {}
    for m, bundle in models.items():
        p = float(bundle["model"].predict_proba(X)[0, 1])
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
        "features": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in feats.items()},
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

    idx = catalog.quakes_in_window(lat, lon, radius_km, start, end.isoformat())
    if len(idx) == 0:
        return {"count": 0, "quakes": []}

    quakes = []
    for i in idx[-500:]:
        quakes.append({
            "time": str(catalog.times_ns[i]),
            "lat": float(catalog.lats[i]),
            "lon": float(catalog.lons[i]),
            "mag": float(catalog.mags[i]),
            "depth": float(catalog.depths[i]),
        })
    return {"count": len(quakes), "quakes": quakes}


if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")