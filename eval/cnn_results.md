## Performanță CNN (EuroSAT, holdout 10%)

Test set: 2,700 imagini (split deterministic seed=42, identic la antrenare).

| Model | Test accuracy | Comentariu |
|---|---:|---|
| Random predictor | 0.1015 | 10 clase echilibrate → ~10% |
| Majority class (`Residential`) | 0.1163 | prezice mereu clasa dominantă |
| SmallCNN (3 conv blocks, 8 epoci, fără transfer learning) | 0.8874 | arhitectura inițială |
| **MobileNetV3-Small + transfer learning + augmentări** (curent) | **0.9789** | 5 epoci head warmup + 10 epoci fine-tune |

**Ablation:** transfer learning + arhitectură mai mare → **+9.1 puncte procentuale accuracy** față de SmallCNN antrenat from-scratch.


### Raport pe clase (model curent)

```
                      precision    recall  f1-score   support

          AnnualCrop      0.967     0.967     0.967       300
              Forest      0.990     0.990     0.990       297
HerbaceousVegetation      0.973     0.970     0.972       302
             Highway      0.974     0.985     0.980       269
          Industrial      0.996     0.977     0.986       257
             Pasture      0.953     0.973     0.963       187
       PermanentCrop      0.992     0.969     0.980       254
         Residential      0.984     0.997     0.991       314
               River      0.960     0.976     0.968       245
             SeaLake      0.993     0.982     0.987       275

            accuracy                          0.979      2700
           macro avg      0.978     0.978     0.978      2700
        weighted avg      0.979     0.979     0.979      2700

```
