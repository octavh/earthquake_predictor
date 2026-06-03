# Platformă de Prognoză a Cutremurelor

Hartă web care arată riscul de cutremur pentru orice locație. Folosește două modele AI: unul care studiază cutremurele trecute și altul care analizează imagini din sateliți pentru a vedea cât de populate sunt zonele.

## Ce face acest proiect

Apasă pe orice locație de pe harta lumii și vezi:

1. Probabilitatea unui cutremur de magnitudine M≥3 / M≥4 / M≥5 / M≥6 / M≥7 într-o rază de 100 km în următoarele 30 de zile (modelul de prognoză LightGBM)
2. Gradul de vulnerabilitate de folosire a terenului in raza selectata
3. Un scor combinat de risc = probabilitatea cutremurului × cât de populate sunt zonele

## Surse de date

| Sursă | Conținut | Licență | Citare |
|---|---|---|---|
| USGS Earthquake Catalog | Catalog global de cutremure 1990–2026, ~3.7M evenimente | Domeniu public | https://earthquake.usgs.gov/fdsnws/event/1/ |
| INFP romplus | Catalog regional România (zona Vrancea), ~50k evenimente | Domeniu public | https://www.infp.ro/data/romplus.txt |
| EuroSAT | Imagini satelitare Sentinel-2, 27.000 imagini etichetate, 10 clase de utilizare a terenului | CC-BY | Helber et al. 2019, https://github.com/phelber/EuroSAT |

## Tehnologii folosite

- **Backend:** Python, FastAPI
- **Modele:** LightGBM (predicții), PyTorch (imagini satelitare) + OpenVINO
- **Frontend:** HTML/CSS/JavaScript, hartă Leaflet
- **Date:** pandas, numpy

## Fișiere și foldere

```
earthquake_predictor/
├── backend/               # Serverul web
├── frontend/              # Pagina hartei
├── scripts/               # Scripturi pentru descărcarea și antrenarea modelelor
├── models/                # Modelele pregătite
├── data/                  # Date descărcate (mare)
├── librarii.txt/          # Librariile utilizate in proiect
└── README.md
```

## Instalare și rulare

### Setup

```cmd
# 1. Descarcă proiectul
git clone https://github.com/octavh/earthquake_predictor.git
cd earthquake_predictor

# 2. Creează mediu virtual
python3 -m venv .venv

# 3. Activeaza mediul virtual
source .venv/bin/activate

# 4. Instalează librarii
pip install -r librarii.txt

# 5. Descarca datele (aprox. 20 min, exista feature-uri care utilizeaza aceste date)
# Fisierele cu date nu au fost adaugate pe Github, deoarece intrec limita de spatiu pentru fisiere.
python3 scripts/model1/download_data.py
python3 scripts/model2/download_data.py

# 6. Pornește serverul
uvicorn backend.main:app --reload
```

### Cum se folosește

Deschide: http://127.0.0.1:8000/app/

Apasă pe o locație de pe hartă și vezi:
- Probabilitatea de cutremur
- Gradul de vulnerabilitate de folosire a terenului
- Indicele de risc total

## Rezultate și evaluare

### Modelele tabulare (LightGBM)

Test set: 82.301 înregistrări cu `prediction_date >= 2020-01-01` din `training_set.csv` (split temporal pentru a evita scurgeri de date).

| Prag | Rata pozitivă | Brier (LGBM) | Brier (baseline constant) | LogLoss (LGBM) | LogLoss (baseline) | ROC-AUC (LGBM) |
|---|---:|---:|---:|---:|---:|---:|
| M≥3 | 0.700 | **0.1389** | 0.2102 | **0.4277** | 0.6112 | **0.871** |
| M≥4 | 0.274 | **0.1773** | 0.1990 | **0.5329** | 0.5873 | **0.756** |
| M≥5 | 0.047 | 0.1263 | **0.0443** | 0.3903 | **0.1880** | **0.797** |
| M≥6 | 0.006 | 0.0509 | **0.0055** | 0.1537 | **0.0345** | **0.804** |
| M≥7 | 0.0005 | 0.1183 | **0.0005** | 1.8199 | **0.0041** | **0.649** |

**Baseline:** predictor constant care prezice rata empirică a clasei pozitive (AUC nedefinit — predicție constantă).

**Interpretare:** la pragurile dese (M≥3, M≥4) modelul bate baseline-ul pe toți indicatorii. La pragurile foarte rare (M≥5, 6, 7) baseline-ul constant calibrat are Brier/LogLoss mai mici pentru că predicția "rata empirică pentru toți" e bine calibrată când clasa pozitivă apare în <5% din cazuri — totuși baseline-ul nu discriminează deloc (AUC = 0.5), pe când modelul atinge AUC 0.65–0.80. Este tradeoff-ul clasic calibrare vs discriminare; modelul rămâne util pentru ranking-ul locațiilor după risc.

### Modelul CNN (EuroSAT, holdout 10% = 2700 imagini)

| Model | Test accuracy | Comentariu |
|---|---:|---|
| Random predictor | 0.1015 | 10 clase ~echilibrate → ~10% |
| Majority class (`Residential`) | 0.1163 | prezice mereu clasa dominantă |
| SmallCNN custom (3 conv blocks, 8 epoci, fără transfer learning, 64×64) | 0.8874 | arhitectura inițială |
| **MobileNetV3-Small + transfer learning + augmentări + 224×224** (curent) | **0.9789** | 5 epoci head warmup + 10 epoci fine-tune cosine LR |

**Ablation Study:** trecerea de la un CNN simplu la transfer learning cu MobileNetV3-Small pretrenat pe ImageNet (cu fine-tune two-phase, augmentări și input 224×224) aduce **+9.1 puncte procentuale accuracy** pe același test split (seed=42).

Raport pe clase (model curent, macro-F1 = 0.978):

| Clasă | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| AnnualCrop | 0.967 | 0.967 | 0.967 | 300 |
| Forest | 0.990 | 0.990 | 0.990 | 297 |
| HerbaceousVegetation | 0.973 | 0.970 | 0.972 | 302 |
| Highway | 0.974 | 0.985 | 0.980 | 269 |
| Industrial | 0.996 | 0.977 | 0.986 | 257 |
| Pasture | 0.953 | 0.973 | 0.963 | 187 |
| PermanentCrop | 0.992 | 0.969 | 0.980 | 254 |
| Residential | 0.984 | 0.997 | 0.991 | 314 |
| River | 0.960 | 0.976 | 0.968 | 245 |
| SeaLake | 0.993 | 0.982 | 0.987 | 275 |

### Vizualizări de erori

- [evaluation/cnn_confusion_matrix.png](eval/cnn_confusion_matrix.png) — matrice de confuzie completă. Erorile dominante sunt confuzii semantic apropiate: `River` ↔ `Highway` (5 cazuri — forme liniare similare în imagini satelitare), `HerbaceousVegetation` ↔ `Pasture` (5 cazuri — texturi vegetale aproape identice), `Industrial` ↔ `Residential` (5 cazuri — acoperișuri urbane similare).
- [evaluation/cnn_misclassified.png](eval/cnn_misclassified.png) — galerie cu 16 din cele 57 imagini clasificate greșit; confirmă vizual că majoritatea erorilor sunt cazuri genuin ambigue chiar și pentru un observator uman.

## API

Serverul oferă aceste adrese:
- `/app/` — harta web
- `/forecast` — calculează riscul pentru o locație
- `/docs` — toate metodele disponibile
