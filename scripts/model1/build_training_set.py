import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.features import CatalogIndex

CATALOG = Path(__file__).parent.parent.parent / "data" / "model1" / "earthquakes.csv"
OUTPUT = Path(__file__).parent.parent.parent / "data" / "model1" / "training_set.csv"

N_TRAIN = 500_000
SEED = 42

def main():
    idx = CatalogIndex(CATALOG)

    rng = np.random.default_rng(SEED)
    n_catalog = idx.n

    sampled_quake_idx = rng.choice(n_catalog, size=N_TRAIN, replace=True)
    sampled_lats = idx.lats[sampled_quake_idx]
    sampled_lons = idx.lons[sampled_quake_idx]
    jitter_lat = rng.uniform(-0.5, 0.5, N_TRAIN)
    jitter_lon = rng.uniform(-0.5, 0.5, N_TRAIN)
    final_lats = sampled_lats + jitter_lat
    final_lons = ((sampled_lons + jitter_lon + 180) % 360) - 180

    start_date = pd.Timestamp("1995-01-01")
    end_date = pd.Timestamp("2024-12-01")
    days_span = (end_date - start_date).days
    sampled_offsets = rng.integers(0, days_span, N_TRAIN)
    sampled_dates = [start_date + timedelta(days=int(d)) for d in sampled_offsets]

    print(f"\nSampled {N_TRAIN:,} (location, date) pairs")
    print(f"Generating features and labels...")

    rows = []
    for i in tqdm(range(N_TRAIN), desc="Building training set"):
        lat = float(final_lats[i])
        lon = float(final_lons[i])
        pred_date = sampled_dates[i].isoformat()

        feats = idx.compute_features(lat, lon, pred_date)
        labs = idx.compute_labels(lat, lon, pred_date)
        rows.append({**feats, **labs, "prediction_date": sampled_dates[i]})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)

    print(f"\nSaved {len(df):,} rows × {len(df.columns)} columns to {OUTPUT}")
    print(f"\nLabel positive rates:")
    for col in ["label_m3", "label_m4", "label_m5", "label_m6", "label_m7"]:
        rate = df[col].mean()
        print(f"  {col}: {rate:.2%}")


if __name__ == "__main__":
    main()