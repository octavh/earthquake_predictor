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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

ROOT = Path(__file__).parent.parent.parent
TRAINING_SET = ROOT / "data" / "model1" / "training_set.csv"
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
SEED = 42

PARAM_DIST = {
    "num_leaves": [15, 31, 63],
    "min_child_samples": [20, 50, 100],
    "learning_rate": [0.05, 0.1],
}
CV_SPLITS = 3
SEARCH_ITER = 6
SEARCH_ESTIMATORS = 200
FINAL_ESTIMATORS = 400


def main():
    print("Loading training set...")
    df = pd.read_csv(TRAINING_SET, parse_dates=["prediction_date"])
    df = df.sort_values("prediction_date").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows")

    train_mask = df["prediction_date"] < SPLIT_DATE
    test_mask = ~train_mask
    print(f"  Train (pre-{SPLIT_DATE}): {train_mask.sum():,}")
    print(f"  Test  (post-{SPLIT_DATE}): {test_mask.sum():,}")

    X_train_full = df.loc[train_mask, FEATURE_COLS].reset_index(drop=True)
    X_test = df.loc[test_mask, FEATURE_COLS].reset_index(drop=True)

    n_train = len(X_train_full)
    val_cut = int(n_train * 0.9)

    summary = {}

    for m in THRESHOLDS:
        label_col = f"label_m{m}"
        y_train_full = df.loc[train_mask, label_col].reset_index(drop=True)
        y_test = df.loc[test_mask, label_col].reset_index(drop=True)

        n_pos = int(y_train_full.sum())
        n_neg = int(len(y_train_full) - n_pos)
        if n_pos < 10:
            print(f"\n[M>={m}] SKIPPED -- only {n_pos} positives in training set")
            continue

        scale_pos = n_neg / n_pos
        print(f"\n[M>={m}]")
        print(f"  train positives: {n_pos:,} ({n_pos/len(y_train_full):.2%})")
        print(f"  test  positives: {int(y_test.sum()):,} ({y_test.mean():.2%})")
        print(f"  scale_pos_weight: {scale_pos:.1f}")

        base_estimator = lgb.LGBMClassifier(
            n_estimators=SEARCH_ESTIMATORS,
            scale_pos_weight=scale_pos,
            random_state=SEED,
            verbose=-1,
        )
        search = RandomizedSearchCV(
            base_estimator,
            PARAM_DIST,
            n_iter=SEARCH_ITER,
            cv=TimeSeriesSplit(n_splits=CV_SPLITS),
            scoring="roc_auc",
            random_state=SEED,
            n_jobs=-1,
            refit=False,
            verbose=0,
        )
        print(f"  Hyperparameter search: {SEARCH_ITER} configs x TimeSeriesSplit({CV_SPLITS})...")
        search.fit(X_train_full, y_train_full)
        best_params = search.best_params_
        cv_mean = float(search.cv_results_["mean_test_score"][search.best_index_])
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        print(f"  best params: {best_params}")
        print(f"  CV ROC-AUC (mean +/- std): {cv_mean:.4f} +/- {cv_std:.4f}")

        X_tr = X_train_full.iloc[:val_cut]
        y_tr = y_train_full.iloc[:val_cut]
        X_val = X_train_full.iloc[val_cut:]
        y_val = y_train_full.iloc[val_cut:]

        final_model = lgb.LGBMClassifier(
            n_estimators=FINAL_ESTIMATORS,
            scale_pos_weight=scale_pos,
            random_state=SEED,
            verbose=-1,
            **best_params,
        )
        final_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        best_iter = final_model.booster_.best_iteration
        print(f"  early stopping at iteration {best_iter}")

        proba = final_model.predict_proba(X_test)[:, 1]
        bs = brier_score_loss(y_test, proba)
        ll = log_loss(y_test, proba, labels=[0, 1])
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")

        print(f"  Holdout Brier:    {bs:.4f}")
        print(f"  Holdout LogLoss:  {ll:.4f}")
        print(f"  Holdout ROC-AUC:  {auc:.4f}")

        metrics = {
            "brier": bs,
            "log_loss": ll,
            "auc": auc,
            "cv_auc_mean": cv_mean,
            "cv_auc_std": cv_std,
            "best_params": best_params,
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "scale_pos_weight": scale_pos,
        }
        summary[m] = metrics

        model_path = MODELS_DIR / f"lgbm_m{m}.pkl"
        joblib.dump({
            "model": final_model,
            "feature_cols": FEATURE_COLS,
            "threshold": m,
            "metrics": metrics,
        }, model_path)
        print(f"  Saved {model_path.name}")

    print("\n=== SUMMARY ===")
    header = f"{'Threshold':<10} {'CV AUC':>20} {'Brier':>8} {'LogLoss':>9} {'AUC':>8}"
    print(header)
    for m, mt in summary.items():
        cv = f"{mt['cv_auc_mean']:.3f} +/- {mt['cv_auc_std']:.3f}"
        print(f"M>={m:<8} {cv:>20} {mt['brier']:>8.4f} {mt['log_loss']:>9.4f} {mt['auc']:>8.4f}")
        print(f"            best_params={mt['best_params']}  early_stop={mt['best_iteration']}")


if __name__ == "__main__":
    main()
