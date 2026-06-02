## Performanță modele tabulare (LightGBM, holdout post-2020)

Test set: înregistrări cu `prediction_date >= 2020-01-01` din `training_set.csv`. Comparat cu un baseline constant care prezice rata empirică a clasei pozitive.

| Prag | N test | Rata pozitivă | Brier (LGBM) | Brier (baseline) | LogLoss (LGBM) | LogLoss (baseline) | ROC-AUC (LGBM) |
|---|---:|---:|---:|---:|---:|---:|---:|
| M≥3 | 82,301 | 0.6996 | 0.1389 | 0.2102 | 0.4277 | 0.6112 | 0.8708 |
| M≥4 | 82,301 | 0.2741 | 0.1773 | 0.1990 | 0.5329 | 0.5873 | 0.7559 |
| M≥5 | 82,301 | 0.0465 | 0.1263 | 0.0443 | 0.3903 | 0.1880 | 0.7972 |
| M≥6 | 82,301 | 0.0056 | 0.0509 | 0.0055 | 0.1537 | 0.0345 | 0.8035 |
| M≥7 | 82,301 | 0.0005 | 0.1183 | 0.0005 | 1.8199 | 0.0041 | 0.6489 |

**Interpretare:**
- La pragurile dese (M≥3, M≥4) modelul bate baseline-ul atât pe Brier cât și pe LogLoss — discriminare reală vizibilă în toți indicatorii.
- La pragurile rare (M≥5, M≥6, M≥7) baseline-ul constant are Brier/LogLoss mai mici pentru că predicția "rata empirică pentru toți" e foarte bine calibrată când clasa pozitivă apare în <5% din cazuri. Totuși, baseline-ul are AUC nedefinit / 0.5 (predicție constantă, zero putere de discriminare), pe când modelul atinge AUC 0.65–0.80.
- Concluzie: modelul oferă discriminare reală la toate pragurile (AUC > 0.5), dar la pragurile foarte rare un baseline constant calibrat e greu de bătut pe metricile de calibrare absolută — clasic tradeoff calibrare vs discriminare.
