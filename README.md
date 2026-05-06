# Platformă de Prognoză a Cutremurelor

Instrument de prognoză probabilistă a hazardului seismic care combină un model ML pe date tabulare antrenat pe cataloage globale de cutremure cu un CNN pe imagini satelitare pentru analiza expunerii.

## Ce face acest proiect

Apasă pe orice locație de pe harta lumii și vezi:

1. Probabilitatea unui cutremur de magnitudine M≥3 / M≥4 / M≥5 / M≥6 / M≥7 într-o rază de 100 km în următoarele 30 de zile (modelul de prognoză LightGBM)
2. Categoria de utilizare a terenului din imagini satelitare (clasificator CNN)
3. Un scor combinat de risc = probabilitatea hazardului × proxy de expunere

Aplicația se concentrează pe zona seismică Vrancea din România ca principal context de problemă comunitară, rămânând în același timp funcțională la nivel global.

## Precizare importantă

Acesta este un instrument de **prognoză**, nu de predicție. Predicția deterministă a cutremurelor (timp / loc / magnitudine specifice) nu este în prezent posibilă științific. Acest proiect generează probabilități calibrate pe baza pattern-urilor istorice — nu predicții ale unor evenimente specifice.

## Surse de date

| Sursă | Conținut | Licență | Citare |
|---|---|---|---|
| USGS Earthquake Catalog | Catalog global de cutremure 1990–2026, ~3.7M evenimente | Domeniu public | https://earthquake.usgs.gov/fdsnws/event/1/ |
| INFP romplus | Catalog regional România (zona Vrancea), ~50k evenimente | Public | https://www.infp.ro/data/romplus.txt |
| EuroSAT | Imagini satelitare Sentinel-2, 27.000 imagini etichetate, 10 clase de utilizare a terenului | CC-BY | Helber et al. 2019, https://github.com/phelber/EuroSAT |
| Esri World Imagery | Plăci satelitare în timp real pentru inferență | Gratuit pentru utilizare necomercială | https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery |

Toate dataseturile sunt publice și citate corespunzător. Nu sunt colectate sau folosite date personale. Cataloagele de cutremure sunt bazate pe măsurători instrumentale, nu pe observații umane, deci nu există probleme de bias uman.

## Stack tehnologic

### Backend
- Python 3.11
- FastAPI (framework web)
- uvicorn (server ASGI)
- python-multipart (suport upload formulare)

### ML / date
- LightGBM (modele de prognoză tabulare)
- PyTorch + torchvision (antrenare CNN)
- scikit-learn (metrici de evaluare)
- pandas, numpy (procesare date)
- joblib (serializare modele LightGBM)
- Pillow (procesare imagini)
- requests (descărcări HTTP, fetch plăci satelitare)
- tqdm (bare de progres în timpul generării setului de antrenare)

### Inferență / optimizare
- Intel OpenVINO (accelerare inferență CNN; cerință a competiției)
- onnxmltools, onnxconverter-common, onnxscript (pipeline export ONNX)

### Frontend
- HTML/CSS/JavaScript vanilla (fără framework)
- Leaflet 1.9.4 (hartă interactivă, încărcată din CDN)
- Plăci OpenStreetMap

### Vizualizare (pentru slide-uri și analiză)
- matplotlib (plot-uri: histogramă magnitudini, scatter mondial, matrice de confuziei, evenimente pe an)

## Structura repository-ului

```
earthquake_predictor/
├── backend/
│   ├── __init__.py
│   ├── features.py          # CatalogIndex, feature engineering, LandUseClassifier
│   └── main.py              # Aplicația FastAPI, toate endpoint-urile
├── frontend/
│   └── index.html           # Hartă Leaflet + sidebar
├── scripts/
│   ├── download_data.py             # Descarcă cataloagele USGS + INFP, le combină și le curăță
│   ├── download_data_m2.py          # Descarcă datasetul de imagini EuroSAT
│   ├── build_training_set.py        # Generează training_set.csv (1M rânduri)
│   ├── train_tabular.py             # Antrenează 5 modele LightGBM
│   ├── train_cnn.py                 # Antrenează CNN-ul satelitar
│   ├── convert_to_openvino.py       # Exportă modelele în ONNX și OpenVINO IR
│   ├── make_slide4_plots.py         # Generează plot-uri de explorare pentru prezentare
│   ├── make_confusion_matrix.py     # Generează matricea de confuziei a CNN
│   ├── count_high_mags.py           # Diagnostic pentru numărul evenimentelor de magnitudine mare
│   └── check_bucharest.py           # Verificare sanity pentru feature engineering
├── models/
│   ├── lgbm_m{3,4,5,6,7}.pkl        # Modele LightGBM antrenate
│   ├── lgbm_m{3,4,5,6,7}.onnx       # Exporturi ONNX (nu folosite la runtime)
│   ├── cnn_eurosat.pth              # CNN PyTorch
│   ├── cnn_eurosat.onnx             # Reprezentare intermediară CNN
│   ├── cnn_eurosat.xml              # OpenVINO IR (folosit pentru inferență)
│   └── cnn_eurosat.bin              # Greutățile OpenVINO IR
├── plots/                           # Vizualizări pentru prezentare
├── data/                            # Date descărcate (gitignored, regenerabile)
├── requirements.txt
└── README.md
```

## Cum se rulează pe calculatorul tău

### Librarii necesare

- fastapi
- uvicorn[standard]
- python-multipart
- lightgbm
- torch
- torchvision
- scikit-learn
- pandas
- numpy
- joblib
- Pillow
- requests
- tqdm
- matplotlib
- openvino
- onnx
- onnxmltools
- onnxconverter-common
- onnxscript

### Cerințe prealabile

- Python 3.11 (alte versiuni 3.x ar putea funcționa dar nu au fost testate)
- git
- ~5 GB spațiu liber pe disc (pentru catalogul de cutremure și EuroSAT)
- macOS, Linux sau Windows
- (Opțional, doar pe macOS) Homebrew pentru a instala libomp pentru LightGBM:

```bash
brew install libomp
```

### Setup pas cu pas

```bash
# 1. Clonează repository-ul
git clone https://github.com/[username-ul-tău]/earthquake_predictor.git
cd earthquake_predictor

# 2. Creează și activează un mediu virtual
python3.11 -m venv .venv
source .venv/bin/activate           # macOS/Linux
# .venv\Scripts\activate            # Windows

# 3. Instalează dependențele Python
pip install --upgrade pip
pip install -r requirements.txt

# 4. Descarcă catalogul de cutremure (~30 min, o singură dată)
python scripts/download_data.py

# 5. Descarcă imaginile satelitare EuroSAT (~5-15 min, o singură dată, ~2 GB)
python scripts/download_data_m2.py

# 6. Construiește setul de antrenare pentru modelul LightGBM (~30-60 min)
python scripts/build_training_set.py

# 7. Antrenează modelele de prognoză LightGBM (~5-15 min)
python scripts/train_tabular.py

# 8. Antrenează CNN-ul satelitar (~30-45 min pe M3 Pro)
python scripts/train_cnn.py

# 9. Convertește modelele în OpenVINO IR (~1 min)
python scripts/convert_to_openvino.py

# 10. Pornește serverul web
uvicorn backend.main:app --reload
```

### Deschide aplicația

Odată ce serverul rulează, deschide în browser:

- http://127.0.0.1:8000/app/ — harta interactivă
- http://127.0.0.1:8000/docs — documentație API generată automat
- http://127.0.0.1:8000/ — verificare stare server

Apasă oriunde pe hartă. Sidebar-ul se actualizează cu probabilitățile de hazard, clasificarea utilizării terenului și scorul combinat de risc pentru acea locație.

### Sărirea peste re-antrenare (setup mai rapid)

Fișierele modelelor antrenate sunt incluse în `models/` (mici, ~10 MB total). Dacă vrei doar să faci demo aplicației, poți sări complet peste pașii 4–9. Doar instalează dependențele și rulează:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Catalogul va fi totuși necesar pentru endpoint-ul `/forecast` ca să calculeze feature-urile. Poți fie să rulezi pasul 4 pentru a descărca catalogul complet (~30 min), fie să folosești un eșantion mai mic pentru testare.

## Endpoint-uri API

| Endpoint | Metodă | Scop |
|---|---|---|
| `/` | GET | Verificare stare server, listează modelele încărcate |
| `/forecast?lat=&lon=&days=&radius_km=` | GET | Prognoză probabilistă pentru o locație |
| `/recent-quakes?lat=&lon=&radius_km=&days=` | GET | Cutremure recente lângă o locație |
| `/vulnerability?lat=&lon=&zoom=` | GET | Preia o placă satelitară și rulează inferența CNN |
| `/classify-image` | POST | Upload o imagine și rulează inferența CNN |
| `/app/` | GET | Harta web interactivă |
| `/docs` | GET | Documentația Swagger / OpenAPI |

## Detalii despre modele

### Modelul de prognoză (LightGBM)
- **Arhitectură:** Decision trees cu gradient boosting, 5 clasificatori binari (unul per prag de magnitudine)
- **Set de antrenare:** 1 milion de rânduri, split temporal (antrenare pre-2020, testare post-2020)
- **Feature-uri (17):** Numărul de cutremure locale (30d, 90d, 365d, 10y), b-value, zile de la ultimul eveniment semnificativ, adâncimea medie, distanța la limita plăcii tectonice, feature-uri de context regional (evenimente în inelul 100-500 km în jurul locației)
- **ROC-AUC pe set de testare:** 0.872 (M≥3), 0.767 (M≥4), 0.792 (M≥5), 0.714 (M≥6), 0.575 (M≥7)

### CNN pentru utilizarea terenului
- **Arhitectură:** CNN custom mic (3 conv + 2 fully connected, ~270k parametri)
- **Antrenat pe:** EuroSAT, 27.000 imagini satelitare, 10 clase
- **Acuratețe pe set de testare:** 89% pe 2.700 imagini held-out
- **Runtime de inferență:** Intel OpenVINO (optimizat pentru CPU)

### Scor combinat de risc

```
Risc = P(M≥5) × Expunere
```

Unde Expunerea este o sumă ponderată a probabilităților claselor CNN înmulțite cu scoruri hardcoded de vulnerabilitate per clasă (Rezidențial = 95, Industrial = 85, Pădure = 10, Mare/Lac = 0, etc.). Acesta este un proxy aproximativ — evaluarea completă a vulnerabilității necesită date la nivel de clădire la care nu am avut acces.