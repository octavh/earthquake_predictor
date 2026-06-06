from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
MC = 4.5
RADIUS_KM = 100
EXTENDED_RADIUS_KM = 500

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

    def _find_seismic_center(self, center_lat, center_lon, prediction_date, days_back=7300):
        """Find distance to main seismic center (cluster of earthquakes in last 20 years)."""
        pred_ns = np.datetime64(prediction_date, "ns")
        cutoff_ns = np.datetime64(pd.Timestamp(prediction_date) - timedelta(days=days_back), "ns")
        time_mask = (self.times_ns < pred_ns) & (self.times_ns >= cutoff_ns)

        if not time_mask.any():
            return 9999.0

        idx = np.where(time_mask)[0]

        mags = self.mags[idx]
        lats = self.lats[idx]
        lons = self.lons[idx]

        weights = 10.0 ** (mags - 3.0)
        center_lat_cluster = np.average(lats, weights=weights)
        center_lon_cluster = np.average(lons, weights=weights)

        dist = haversine(center_lat, center_lon, center_lat_cluster, center_lon_cluster)
        return float(dist)

    def _regional_context(self, center_lat, center_lon, prediction_date, radius_km):
        pred_ns = np.datetime64(prediction_date, "ns")
        cutoff_20y_ns = np.datetime64(pd.Timestamp(prediction_date) - timedelta(days=7300), "ns")
        cutoff_10y_ns = np.datetime64(pd.Timestamp(prediction_date) - timedelta(days=3650), "ns")

        time_mask_20y = (self.times_ns < pred_ns) & (self.times_ns >= cutoff_20y_ns)

        n_m3_20y = 0
        n_m5_20y = 0
        n_m6_20y = 0
        n_m7_20y = 0
        n_m5_ring_20y = 0
        n_m6_ring_20y = 0
        dist_to_m5_20y = float(EXTENDED_RADIUS_KM)
        n_m5_ring_10y = 0
        n_m6_ring_10y = 0
        dist_to_m5_10y = float(EXTENDED_RADIUS_KM)
        seismic_center_dist = self._find_seismic_center(center_lat, center_lon, prediction_date, 7300)

        if time_mask_20y.any():
            recent_idx = np.where(time_mask_20y)[0]
            recent_times = self.times_ns[recent_idx]
            recent_dist = haversine(center_lat, center_lon, self.lats[recent_idx], self.lons[recent_idx])
            recent_mags = self.mags[recent_idx]

            local_mask = recent_dist <= radius_km
            local_mags = recent_mags[local_mask]
            n_m3_20y = int((local_mags >= 3.0).sum())
            n_m5_20y = int((local_mags >= 5.0).sum())
            n_m6_20y = int((local_mags >= 6.0).sum())
            n_m7_20y = int((local_mags >= 7.0).sum())

            in_ring = (recent_dist > radius_km) & (recent_dist <= EXTENDED_RADIUS_KM)
            ring_mags = recent_mags[in_ring]
            ring_dists = recent_dist[in_ring]

            n_m5_ring_20y = int((ring_mags >= 5.0).sum())
            n_m6_ring_20y = int((ring_mags >= 6.0).sum())

            local_m5_mask = (recent_dist <= radius_km) & (recent_mags >= 5.0)
            if local_m5_mask.any():
                dist_to_m5_20y = 0.0
            elif (ring_mags >= 5.0).any():
                dist_to_m5_20y = float(ring_dists[ring_mags >= 5.0].min())
            else:
                dist_to_m5_20y = float(EXTENDED_RADIUS_KM)

            mask_10y = recent_times >= cutoff_10y_ns
            if mask_10y.any():
                dist_10y = recent_dist[mask_10y]
                mags_10y = recent_mags[mask_10y]
                in_ring_10y = (dist_10y > radius_km) & (dist_10y <= EXTENDED_RADIUS_KM)
                ring_mags_10y = mags_10y[in_ring_10y]
                ring_dists_10y = dist_10y[in_ring_10y]

                n_m5_ring_10y = int((ring_mags_10y >= 5.0).sum())
                n_m6_ring_10y = int((ring_mags_10y >= 6.0).sum())

                local_m5_mask_10y = (dist_10y <= radius_km) & (mags_10y >= 5.0)
                if local_m5_mask_10y.any():
                    dist_to_m5_10y = 0.0
                elif (ring_mags_10y >= 5.0).any():
                    dist_to_m5_10y = float(ring_dists_10y[ring_mags_10y >= 5.0].min())
                else:
                    dist_to_m5_10y = float(EXTENDED_RADIUS_KM)

        m6_mask = self.mags >= 6.0
        if m6_mask.any():
            m6_dist = haversine(center_lat, center_lon, self.lats[m6_mask], self.lons[m6_mask])
            dist_to_m6_ever = float(m6_dist.min())
        else:
            dist_to_m6_ever = 9999.0

        return {
            "n_m3_20y": n_m3_20y,
            "n_m5_20y": n_m5_20y,
            "n_m6_20y": n_m6_20y,
            "n_m7_20y": n_m7_20y,
            "n_m5_ring_20y": n_m5_ring_20y,
            "n_m6_ring_20y": n_m6_ring_20y,
            "dist_to_nearest_m5_20y": dist_to_m5_20y,
            "n_m5_ring_10y": n_m5_ring_10y,
            "n_m6_ring_10y": n_m6_ring_10y,
            "dist_to_nearest_m5_10y": dist_to_m5_10y,
            "dist_to_nearest_m6_ever": dist_to_m6_ever,
            "seismic_center_dist_20y": seismic_center_dist,
        }

    def _compute_adaptive_depth(self, center_lat, center_lon, prediction_date, initial_radius_km=100):
        """Compute mean depth, widening radius if needed to get values."""
        pred_ns = np.datetime64(prediction_date, "ns")
        cutoff_3y_ns = np.datetime64(pd.Timestamp(prediction_date) - timedelta(days=1095), "ns")
        time_mask = (self.times_ns < pred_ns) & (self.times_ns >= cutoff_3y_ns)

        if not time_mask.any():
            return np.nan

        search_radii = [initial_radius_km, 200, 300, 500, 1000]
        for r in search_radii:
            idx = self.quakes_in_window(center_lat, center_lon, r,
                                       (pd.Timestamp(prediction_date) - timedelta(days=1095)).isoformat(),
                                       prediction_date)
            if len(idx) >= 5:
                return float(self.depths[idx].mean())

        return np.nan

    def compute_features(self, center_lat, center_lon, prediction_date, radius_km=RADIUS_KM):
        pred_dt = pd.Timestamp(prediction_date)

        history_idx = self.quakes_in_window(
            center_lat, center_lon, radius_km, "1990-01-01", prediction_date,
        )

        ctx = self._regional_context(center_lat, center_lon, prediction_date, radius_km)

        if len(history_idx) == 0:
            return {
                "lat": center_lat, "lon": center_lon,
                "n_30d": 0, "n_90d": 0, "n_365d": 0, "n_3650d": 0,
                "max_mag_365d": 0.0, "mean_mag_365d": 0.0,
                "days_since_m4": 9999.0, "days_since_m5": 9999.0,
                "b_value_10y": np.nan, "a_value_10y": np.nan,
                "mean_depth_365d": np.nan,
                "dist_to_plate": float(distance_to_plate_boundary(center_lat, center_lon)),
                **ctx,
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

        mags_10y = history_mags[last_3650]
        bv = b_value(mags_10y, mc=3.0)
        if not np.isnan(bv):
            n_at_mc = (mags_10y >= 3.0).sum()
            av = float(np.log10(n_at_mc) + bv * 3.0) if n_at_mc > 0 else np.nan
        else:
            av = np.nan

        mean_depth = self._compute_adaptive_depth(center_lat, center_lon, prediction_date, radius_km)

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
            "mean_depth_adaptive": mean_depth,
            "dist_to_plate": float(distance_to_plate_boundary(center_lat, center_lon)),
            **ctx,
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
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

VULNERABILITY_SCORES = {
    "Residential": 100,
    "Industrial": 100,
    "Highway": 80,
    "AnnualCrop": 20,
    "PermanentCrop": 20,
    "Pasture": 10,
    "HerbaceousVegetation": 10,
    "Forest": 10,
    "River": 5,
    "SeaLake": 0,
}


class LandUseModel(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        in_feats = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)


_SmallCNN = LandUseModel


class LandUseClassifier:
    EUROSAT_CLASSES = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    ]

    def __init__(self, model_path: Path):
        model_path = Path(model_path)

        self.classes = self.EUROSAT_CLASSES
        self.input_size = 224
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

        bundle = self._load_bundle(model_path)

        ir_path = model_path.with_suffix(".xml")
        loaded = False
        if ir_path.exists():
            try:
                import openvino as ov
                core = ov.Core()
                self.compiled_model = core.compile_model(str(ir_path), "CPU")
                self.input_layer = next(iter(self.compiled_model.inputs))
                self.output_layer = next(iter(self.compiled_model.outputs))
                self.runtime = "openvino"
                print(f"LandUseClassifier loaded (OpenVINO): {len(self.classes)} classes, input_size={self.input_size}")
                loaded = True
            except Exception as e:
                print(f"OpenVINO load failed ({e}), falling back to PyTorch")

        if not loaded:
            if bundle is None:
                raise RuntimeError(f"No usable land-use model at {model_path} (neither .xml IR nor .pth bundle)")
            self._init_pytorch(bundle)

        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
        ])

    def _load_bundle(self, model_path):
        if model_path.exists():
            try:
                bundle = torch.load(model_path, map_location="cpu", weights_only=False)
                self.classes = bundle.get("classes", self.classes)
                self.input_size = bundle.get("input_size", self.input_size)
                self.normalize_mean = bundle.get("normalize_mean", self.normalize_mean)
                self.normalize_std = bundle.get("normalize_std", self.normalize_std)
                return bundle
            except Exception as e:
                print(f"Warning: could not load bundle from {model_path.name}: {e}")
        return None

    def _init_pytorch(self, bundle):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = LandUseModel(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()
        self.runtime = "pytorch"
        print(f"LandUseClassifier loaded (PyTorch): {len(self.classes)} classes, input_size={self.input_size}")

    def classify_image(self, pil_image: Image.Image) -> dict:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Preprocess to numpy
        tensor = self.transform(pil_image)
        input_array = tensor.numpy()

        if self.runtime == "openvino":
            input_array = np.expand_dims(input_array, 0)
            result = self.compiled_model([input_array])
            logits = result[self.output_layer][0]
            exp_logits = np.exp(logits - logits.max())
            probs = exp_logits / exp_logits.sum()
        else:
            with torch.no_grad():
                tensor_batch = tensor.unsqueeze(0).to(self.device)
                logits = self.model(tensor_batch)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_idx = int(probs.argmax())
        top_class = self.classes[top_idx]
        top_prob = float(probs[top_idx])

        weighted = sum(
            float(probs[i]) * VULNERABILITY_SCORES.get(c, 0)
            for i, c in enumerate(self.classes)
        )

        return {
            "predicted_class": top_class,
            "confidence": round(top_prob, 3),
            "vulnerability_score": round(weighted, 1),
            "all_classes": {c: round(float(probs[i]), 3) for i, c in enumerate(self.classes)},
        }