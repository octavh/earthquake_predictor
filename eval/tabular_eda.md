# EDA — date structurate (catalog seismic)

**Sursa principala:** USGS Earthquake Catalog 1990–2026 (~3.7M evenimente) + INFP romplus (Vrancea, ~50k).

**Training set:** `data/model1/training_set.csv` — 500,000 sample-uri (lat, lon, data) cu 18 feature-uri + 5 etichete (label_m3..m7).

**Construit prin:** sample uniform din catalog + jitter ±0.5° + dată random 1995–2024 + calcul features în jurul fiecărui punct.


## Dezechilibru clase

| Prag | Rata pozitivă | Comentariu |
|---|---:|---|
| M>=3 | 0.6752 (67.52%) | majoritar pozitiv |
| M>=4 | 0.2481 (24.81%) | majoritar negativ |
| M>=5 | 0.0447 (4.47%) | rar |
| M>=6 | 0.0056 (0.56%) | rar |
| M>=7 | 0.0009 (0.09%) | extrem de rar |

Vezi [figures/tabular_class_imbalance.png](figures/tabular_class_imbalance.png).

**Decizie de modelare:** pentru fiecare prag antrenăm un model separat cu `scale_pos_weight = n_neg / n_pos` în LightGBM, astfel încât pierderea să rămână echilibrată chiar și la M>=7 unde clasa pozitivă apare în ~0.05% din cazuri.

## Acoperire geografică

Sample-urile sunt distribuite global (vezi [figures/tabular_geographic_coverage.png](figures/tabular_geographic_coverage.png)), cu densitate vizibil mai mare în zone bine instrumentate seismic (USA West Coast, Japonia, arcul mediteranean, lanțul andin, Indonezia). Sub-acoperirea oceanelor și a Africii sub-sahariene este o limitare moștenită de la catalogul USGS și e discutată la secțiunea Etică din README.

## Distribuții și valori lipsă

Vezi [figures/tabular_feature_distributions.png](figures/tabular_feature_distributions.png) pentru 6 feature-uri reprezentative.


| Feature | NaN | Inf | Strategie |
|---|---:|---:|---|
| `n_30d` | 0 | 0 | `np.nan_to_num(nan=0, posinf=9999, neginf=-9999)` la inferență |
| `n_365d` | 0 | 0 | `np.nan_to_num(nan=0, posinf=9999, neginf=-9999)` la inferență |
| `days_since_m5` | 0 | 0 | `np.nan_to_num(nan=0, posinf=9999, neginf=-9999)` la inferență |
| `b_value_10y` | 28,719 | 0 | `np.nan_to_num(nan=0, posinf=9999, neginf=-9999)` la inferență |
| `dist_to_plate` | 0 | 0 | `np.nan_to_num(nan=0, posinf=9999, neginf=-9999)` la inferență |
| `dist_to_nearest_m5_10y` | 0 | 0 | `np.nan_to_num(nan=0, posinf=9999, neginf=-9999)` la inferență |

**Observații cheie:**
- `days_since_m5` este 9999 pentru locațiile fără nicio mișcare M>=5 înregistrată — codare santinelă pentru "nu s-a întâmplat niciodată".
- `b_value_10y` (panta Gutenberg-Richter) lipsește pentru zonele cu prea puține cutremure pentru un fit credibil; NaN tratat ca 0 în inferență.
- `dist_to_plate` este derivat din shapefile NUVEL — nu are lipsuri.
- Distribuțiile sunt puternic asimetrice (long-tail), motiv pentru care folosim un model bazat pe arbori (LightGBM) care nu cere normalizare.
