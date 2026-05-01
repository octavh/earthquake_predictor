import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    classification_report,
)

ROOT = Path(__file__).parent.parent
TRAINING_SET = ROOT / "data" / "training_set.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

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

THRESHOLDS = [3, 4, 5, 6, 7]
SPLIT_DATE = "2020-01-01"


def main():
    print(f"Loading training set...")
    df = pd.read_csv(TRAINING_SET, parse_dates=["prediction_date"])
    print(f"  Loaded {len(df):,} rows")

    train_mask = df["prediction_date"] < SPLIT_DATE
    test_mask = ~train_mask
    print(f"  Train (pre-{SPLIT_DATE}): {train_mask.sum():,}")
    print(f"  Test  (post-{SPLIT_DATE}): {test_mask.sum():,}")

    X_train = df.loc[train_mask, FEATURE_COLS]
    X_test = df.loc[test_mask, FEATURE_COLS]

    metrics = {}

    for m in THRESHOLDS:
        label_col = f"label_m{m}"
        y_train = df.loc[train_mask, label_col]
        y_test = df.loc[test_mask, label_col]

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_pos < 10:
            print(f"\n[M≥{m}] SKIPPED — only {n_pos} positives in training set")
            continue

        scale_pos = n_neg / n_pos

        print(f"\n[M≥{m}]")
        print(f"  train positives: {n_pos:,} ({n_pos/len(y_train):.2%})")
        print(f"  test  positives: {y_test.sum():,} ({y_test.mean():.2%})")
        print(f"  scale_pos_weight: {scale_pos:.1f}")

        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            scale_pos_weight=scale_pos,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        bs = brier_score_loss(y_test, proba)
        ll = log_loss(y_test, proba, labels=[0, 1])
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")

        print(f"  Brier:    {bs:.4f}")
        print(f"  Log-loss: {ll:.4f}")
        print(f"  ROC-AUC:  {auc:.4f}")

        metrics[m] = {"brier": bs, "log_loss": ll, "auc": auc}

        model_path = MODELS_DIR / f"lgbm_m{m}.pkl"
        joblib.dump({
            "model": model,
            "feature_cols": FEATURE_COLS,
            "threshold": m,
            "metrics": metrics[m],
        }, model_path)
        print(f"  Saved {model_path.name}")

    print("\n=== SUMMARY ===")
    print(f"{'Threshold':<10} {'Brier':>8} {'LogLoss':>9} {'ROC-AUC':>9}")
    for m, mt in metrics.items():
        print(f"M≥{m:<8} {mt['brier']:>8.4f} {mt['log_loss']:>9.4f} {mt['auc']:>9.4f}")


if __name__ == "__main__":
    main()