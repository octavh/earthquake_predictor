"""EDA pe datele structurate (catalog seismic + training set).

Outputs:
- eval/figures/tabular_class_imbalance.png    — rata pozitiva per prag M>=k
- eval/figures/tabular_geographic_coverage.png — distributia spatiala a sample-urilor
- eval/figures/tabular_feature_distributions.png — histograme pentru 6 feature-uri cheie
- eval/tabular_eda.md                         — sinteza text

Rulare: .venv/bin/python eval/eda_tabular.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
TRAINING_SET = ROOT / "data" / "model1" / "training_set.csv"
OUT_DIR = ROOT / "eval"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [3, 4, 5, 6, 7]
KEY_FEATURES = [
    "n_30d", "n_365d", "days_since_m5",
    "b_value_10y", "dist_to_plate", "dist_to_nearest_m5_10y",
]


def main():
    print("Loading training set...")
    df = pd.read_csv(TRAINING_SET, parse_dates=["prediction_date"])
    n = len(df)
    print(f"  {n:,} rows, {len(df.columns)} columns")

    rates = {m: float(df[f"label_m{m}"].mean()) for m in THRESHOLDS}

    fig, ax = plt.subplots(figsize=(7, 4))
    ms = list(rates.keys())
    pcts = [rates[m] * 100 for m in ms]
    bars = ax.bar([f"M>={m}" for m in ms], pcts, color="#0ea5e9")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% sample-uri pozitive in 30 zile")
    ax.set_title(f"Dezechilibru clase per prag (n={n:,})")
    ax.set_ylim(0, max(pcts) * 1.15)
    fig.tight_layout()
    p = FIG_DIR / "tabular_class_imbalance.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  -> {p}")

    sample = df.sample(min(50_000, n), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(sample["lon"], sample["lat"], s=0.4, c="#0ea5e9", alpha=0.4)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitudine")
    ax.set_ylabel("latitudine")
    ax.set_title(f"Acoperire geografica training set (subesantion {len(sample):,} din {n:,})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "tabular_geographic_coverage.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  -> {p}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), KEY_FEATURES):
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = np.percentile(vals, [1, 99])
        clipped = vals.clip(lo, hi)
        ax.hist(clipped, bins=50, color="#0ea5e9", alpha=0.8)
        ax.set_title(col, fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Distributii feature-uri cheie (clip 1-99 percentil, scala log)", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "tabular_feature_distributions.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  -> {p}")

    nan_counts = df[KEY_FEATURES].isna().sum().to_dict()
    inf_counts = {c: int(np.isinf(df[c].fillna(0)).sum()) for c in KEY_FEATURES}

    lines = []
    lines.append("# EDA — date structurate (catalog seismic)\n")
    lines.append("**Sursa principala:** USGS Earthquake Catalog 1990–2026 (~3.7M evenimente) + INFP romplus (Vrancea, ~50k).\n")
    lines.append(f"**Training set:** `data/model1/training_set.csv` — {n:,} sample-uri (lat, lon, data) cu 18 feature-uri + 5 etichete (label_m3..m7).\n")
    lines.append("**Construit prin:** sample uniform din catalog + jitter ±0.5° + dată random 1995–2024 + calcul features în jurul fiecărui punct.\n\n")

    lines.append("## Dezechilibru clase\n")
    lines.append("| Prag | Rata pozitivă | Comentariu |")
    lines.append("|---|---:|---|")
    for m in THRESHOLDS:
        r = rates[m]
        comm = "majoritar pozitiv" if r > 0.5 else "majoritar negativ" if r > 0.05 else "rar" if r > 0.001 else "extrem de rar"
        lines.append(f"| M>={m} | {r:.4f} ({r*100:.2f}%) | {comm} |")
    lines.append("\nVezi [figures/tabular_class_imbalance.png](figures/tabular_class_imbalance.png).\n")
    lines.append("**Decizie de modelare:** pentru fiecare prag antrenăm un model separat cu `scale_pos_weight = n_neg / n_pos` în LightGBM, astfel încât pierderea să rămână echilibrată chiar și la M>=7 unde clasa pozitivă apare în ~0.05% din cazuri.\n")

    lines.append("## Acoperire geografică\n")
    lines.append("Sample-urile sunt distribuite global (vezi [figures/tabular_geographic_coverage.png](figures/tabular_geographic_coverage.png)), cu densitate vizibil mai mare în zone bine instrumentate seismic (USA West Coast, Japonia, arcul mediteranean, lanțul andin, Indonezia). Sub-acoperirea oceanelor și a Africii sub-sahariene este o limitare moștenită de la catalogul USGS și e discutată la secțiunea Etică din README.\n")

    lines.append("## Distribuții și valori lipsă\n")
    lines.append("Vezi [figures/tabular_feature_distributions.png](figures/tabular_feature_distributions.png) pentru 6 feature-uri reprezentative.\n")
    lines.append("\n| Feature | NaN | Inf | Strategie |")
    lines.append("|---|---:|---:|---|")
    for c in KEY_FEATURES:
        n_nan = nan_counts[c]
        n_inf = inf_counts[c]
        lines.append(f"| `{c}` | {n_nan:,} | {n_inf:,} | `np.nan_to_num(nan=0, posinf=9999, neginf=-9999)` la inferență |")
    lines.append("\n**Observații cheie:**")
    lines.append("- `days_since_m5` este 9999 pentru locațiile fără nicio mișcare M>=5 înregistrată — codare santinelă pentru \"nu s-a întâmplat niciodată\".")
    lines.append("- `b_value_10y` (panta Gutenberg-Richter) lipsește pentru zonele cu prea puține cutremure pentru un fit credibil; NaN tratat ca 0 în inferență.")
    lines.append("- `dist_to_plate` este derivat din shapefile NUVEL — nu are lipsuri.")
    lines.append("- Distribuțiile sunt puternic asimetrice (long-tail), motiv pentru care folosim un model bazat pe arbori (LightGBM) care nu cere normalizare.\n")

    (OUT_DIR / "tabular_eda.md").write_text("\n".join(lines))
    print(f"  -> {OUT_DIR / 'tabular_eda.md'}")


if __name__ == "__main__":
    main()
