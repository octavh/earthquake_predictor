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

# 5. Descarca datele (aprox. 10 min, exista feature-uri care utilizeaza aceste date)
# Fisierele cu date nu au fost adaugate pe Github, deoarece intrec limita de spatiu pentru fisiere.
python3 scripts/model1/download_data.py

# 6. Pornește serverul
uvicorn backend.main:app --reload
```

### Cum se folosește

Deschide: http://127.0.0.1:8000/app/

Apasă pe o locație de pe hartă și vezi:
- Probabilitatea de cutremur
- Gradul de vulnerabilitate de folosire a terenului
- Indicele de risc total

## API

Serverul oferă aceste adrese:
- `/app/` — harta web
- `/forecast` — calculează riscul pentru o locație
- `/docs` — toate metodele disponibile
