# EDA — date nestructurate (EuroSAT)

**Sursa:** EuroSAT RGB (Helber et al. 2019) — patch-uri Sentinel-2 de 64x64 px, 10 clase de utilizare a terenului.

**Total:** 27,000 imagini, mediu ~2,700 per clasa. Cea mai mica clasa: `Pasture` (2000), cea mai mare: `AnnualCrop` (3000).

Dataset-ul este suficient de echilibrat încât nu necesită oversampling/class weighting în pierdere — CrossEntropy standard este adecvată.

## Distributie clase

Vezi [figures/cnn_class_distribution.png](figures/cnn_class_distribution.png).


| Clasă | Imagini | % din total |
|---|---:|---:|
| AnnualCrop | 3,000 | 11.1% |
| Forest | 3,000 | 11.1% |
| HerbaceousVegetation | 3,000 | 11.1% |
| Highway | 2,500 | 9.3% |
| Industrial | 2,500 | 9.3% |
| Pasture | 2,000 | 7.4% |
| PermanentCrop | 2,500 | 9.3% |
| Residential | 3,000 | 11.1% |
| River | 2,500 | 9.3% |
| SeaLake | 3,000 | 11.1% |

## Exemple reprezentative

Galeria [figures/cnn_sample_grid.png](figures/cnn_sample_grid.png) arată câte un patch tipic per clasă. Observații vizuale relevante pentru proiectare:

- `River` și `Highway` sunt ambele structuri liniare în imagini satelitare → confuzii frecvente confirmate ulterior în matricea de confuzie.
- `HerbaceousVegetation` și `Pasture` au texturi similare verde-uniform → la fel.
- `Residential` și `Industrial` au pattern-uri de acoperișuri ușor diferite — diferența e subtilă chiar și pentru un observator uman.
- `SeaLake` și `Forest` sunt cele mai bine separabile vizual (omogenitate cromatică distinctă).

Exemple concrete de erori sunt în [cnn_misclassified.png](cnn_misclassified.png).

## Pipeline preprocesare + augmentări

**Inferență (test-time):**

```
Resize(224, 224)  ->  ToTensor()  ->  Normalize(ImageNet mean/std)
```
Upsamplingul 64->224 este necesar pentru backbone-ul MobileNetV3-Small pretrenat pe ImageNet (224 este input-ul standard).


**Augmentări la antrenare (numai pe train, val/test deterministe):**
- `RandomHorizontalFlip()` — imaginile satelitare nu au orientare stânga/dreapta canonică
- `RandomVerticalFlip()` — nici nord/sud (Sentinel-2 capturează la orbite multiple)
- `RandomRotation(15°)` — modelează variațiile de unghi orbital
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` — modelează variabilitatea sezonieră și atmosferică

Normalizarea folosește media și deviația standard de pe ImageNet (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) — obligatoriu pentru ca greutățile pretrenate să primească exact distribuția pe care au fost antrenate.
