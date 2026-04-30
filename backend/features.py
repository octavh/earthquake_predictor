"""Feature engineering for earthquake forecasting."""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
MC = 4.5
RADIUS_KM = 100

PLATE_BOUNDARY_POINTS = np.array([
    [-90, 0], [-60, -75], [-35, -72], [-15, -75], [0, -80], [10, -85],
    [20, -105], [35, -120], [50, -130], [55, -160], [50, 160], [40, 142],
    [35, 140], [30, 132], [25, 125], [15, 120], [0, 130], [-10, 150],
    [-20, 175], [-40, 175], [-55, -70], [-50, -75],
    [60, -30], [40, -30], [20, -45], [0, -25], [-30, -15], [-55, -20],
    [40, -10], [38, 10], [40, 25], [38, 45], [33, 50], [30, 70], [28, 85], [25, 95],
    [-10, 110], [-20, 120], [-30, 135],
])


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def b_value(magnitudes, mc=3.0, min_count=15):
    """Aki maximum-likelihood b-value. Mc=3.0 chosen for ML feature coverage."""
    mags = np.asarray(magnitudes)
    mags = mags[mags >= mc]
    if len(mags) < min_count:
        return np.nan
    return np.log10(np.e) / (mags.mean() - mc)


def distance_to_plate_boundary(lat, lon):
    return haversine(lat, lon, PLATE_BOUNDARY_POINTS[:, 0], PLATE_BOUNDARY_POINTS[:, 1]).min()


class CatalogIndex:
    def __init__(self, catalog_path: Path):
        df = pd.read_csv(catalog_path, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["time", "latitude", "longitude", "mag", "depth"])
        df = df.sort_values("time").reset_index(drop=True)

        self.times_ns = df["time"].values.astype("datetime64[ns]")
        self.lats = df["latitude"].values
        self.lons = df["longitude"].values
        self.mags = df["mag"].values
        self.depths = df["depth"].values
        self.n = len(df)
        print(f"CatalogIndex loaded: {self.n:,} events")

    def quakes_in_window(self, center_lat, center_lon, radius_km, start_time, end_time):
        start_ns = np.datetime64(start_time, "ns")
        end_ns = np.datetime64(end_time, "ns")
        time_mask = (self.times_ns >= start_ns) & (self.times_ns < end_ns)
        if not time_mask.any():
            return np.array([], dtype=np.int64)
        idx = np.where(time_mask)[0]
        d = haversine(center_lat, center_lon, self.lats[idx], self.lons[idx])
        return idx[d <= radius_km]

    def compute_features(self, center_lat, center_lon, prediction_date, radius_km=RADIUS_KM):
        pred_dt = pd.Timestamp(prediction_date)

        history_idx = self.quakes_in_window(
            center_lat, center_lon, radius_km, "1990-01-01", prediction_date,
        )

        if len(history_idx) == 0:
            return {
                "lat": center_lat, "lon": center_lon,
                "n_30d": 0, "n_90d": 0, "n_365d": 0, "n_3650d": 0,
                "max_mag_365d": 0.0, "mean_mag_365d": 0.0,
                "days_since_m4": 9999.0, "days_since_m5": 9999.0,
                "b_value_10y": np.nan, "a_value_10y": np.nan,
                "mean_depth_365d": np.nan,
                "dist_to_plate": float(distance_to_plate_boundary(center_lat, center_lon)),
            }

        history_times = self.times_ns[history_idx]
        history_mags = self.mags[history_idx]
        history_depths = self.depths[history_idx]

        def in_last(n):
            cutoff = np.datetime64(pred_dt - timedelta(days=n), "ns")
            return history_times >= cutoff

        last_30 = in_last(30)
        last_90 = in_last(90)
        last_365 = in_last(365)
        last_3650 = in_last(3650)

        m4_times = history_times[history_mags >= 4.0]
        m5_times = history_times[history_mags >= 5.0]

        def days_since(arr):
            if len(arr) == 0:
                return 9999.0
            return (pred_dt - pd.Timestamp(arr.max())).total_seconds() / 86400.0

        # b-value at mc=3.0 for ML feature coverage; min_count=15 for stability
        mags_10y = history_mags[last_3650]
        bv = b_value(mags_10y, mc=3.0)
        if not np.isnan(bv):
            n_at_mc = (mags_10y >= 3.0).sum()
            av = float(np.log10(n_at_mc) + bv * 3.0) if n_at_mc > 0 else np.nan
        else:
            av = np.nan

        return {
            "lat": center_lat, "lon": center_lon,
            "n_30d": int(last_30.sum()),
            "n_90d": int(last_90.sum()),
            "n_365d": int(last_365.sum()),
            "n_3650d": int(last_3650.sum()),
            "max_mag_365d": float(history_mags[last_365].max()) if last_365.any() else 0.0,
            "mean_mag_365d": float(history_mags[last_365].mean()) if last_365.any() else 0.0,
            "days_since_m4": days_since(m4_times),
            "days_since_m5": days_since(m5_times),
            "b_value_10y": float(bv) if not np.isnan(bv) else np.nan,
            "a_value_10y": av,
            "mean_depth_365d": float(history_depths[last_365].mean()) if last_365.any() else np.nan,
            "dist_to_plate": float(distance_to_plate_boundary(center_lat, center_lon)),
        }

    def compute_labels(self, center_lat, center_lon, prediction_date,
                       radius_km=RADIUS_KM, window_days=30):
        pred_dt = pd.Timestamp(prediction_date)
        end_dt = pred_dt + timedelta(days=window_days)

        future_idx = self.quakes_in_window(
            center_lat, center_lon, radius_km,
            prediction_date, end_dt.isoformat(),
        )

        if len(future_idx) == 0:
            return {f"label_m{m}": 0 for m in [3, 4, 5, 6, 7]}

        future_mags = self.mags[future_idx]
        return {f"label_m{m}": int((future_mags >= m).any()) for m in [3, 4, 5, 6, 7]}