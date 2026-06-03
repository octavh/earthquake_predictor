"""EDA pe datele nestructurate (imagini EuroSAT).

Outputs:
- eval/figures/cnn_class_distribution.png  — numar imagini per clasa
- eval/figures/cnn_sample_grid.png         — grila 2x5 cu cate o imagine reprezentativa per clasa
- eval/cnn_eda.md                          — sinteza text

Rulare: .venv/bin/python eval/eda_images.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "model2"
OUT_DIR = ROOT / "eval"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_class_root(extract_dir: Path) -> Path:
    candidates = list(extract_dir.glob("**/AnnualCrop"))
    if not candidates:
        raise RuntimeError(f"Could not find class folders under {extract_dir}")
    return candidates[0].parent


def main():
    class_root = find_class_root(DATA_ROOT)
    print(f"Loading EuroSAT from {class_root}")
    classes = sorted([p.name for p in class_root.iterdir() if p.is_dir()])
    print(f"  {len(classes)} classes: {classes}")

    counts = {}
    sample_paths = {}
    for c in classes:
        files = sorted((class_root / c).glob("*.jpg"))
        counts[c] = len(files)
        sample_paths[c] = files[0] if files else None
    total = sum(counts.values())
    print(f"  total images: {total:,}")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(classes, [counts[c] for c in classes], color="#0ea5e9")
    for bar, c in zip(bars, classes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(counts[c]), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("numar imagini")
    ax.set_title(f"Distributie clase EuroSAT (n={total:,}, ~echilibrat)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylim(0, max(counts.values()) * 1.15)
    fig.tight_layout()
    p = FIG_DIR / "cnn_class_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  -> {p}")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, c in zip(axes.ravel(), classes):
        img_path = sample_paths[c]
        if img_path is not None and img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax.imshow(np.asarray(img))
        ax.set_title(c, fontsize=10)
        ax.axis("off")
    fig.suptitle("EuroSAT — cate o imagine reprezentativa per clasa (64x64 nativ)", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "cnn_sample_grid.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  -> {p}")

    min_c = min(counts, key=counts.get)
    max_c = max(counts, key=counts.get)
    avg = total // len(classes)

    lines = []
    lines.append("# EDA — date nestructurate (EuroSAT)\n")
    lines.append("**Sursa:** EuroSAT RGB (Helber et al. 2019) — patch-uri Sentinel-2 de 64x64 px, 10 clase de utilizare a terenului.\n")
    lines.append(f"**Total:** {total:,} imagini, mediu ~{avg:,} per clasa. Cea mai mica clasa: `{min_c}` ({counts[min_c]}), cea mai mare: `{max_c}` ({counts[max_c]}).\n")
    lines.append("Dataset-ul este suficient de echilibrat încât nu necesită oversampling/class weighting în pierdere — CrossEntropy standard este adecvată.\n")

    lines.append("## Distributie clase\n")
    lines.append("Vezi [figures/cnn_class_distribution.png](figures/cnn_class_distribution.png).\n")
    lines.append("\n| Clasă | Imagini | % din total |")
    lines.append("|---|---:|---:|")
    for c in classes:
        lines.append(f"| {c} | {counts[c]:,} | {counts[c]/total*100:.1f}% |")
    lines.append("")

    lines.append("## Exemple reprezentative\n")
    lines.append("Galeria [figures/cnn_sample_grid.png](figures/cnn_sample_grid.png) arată câte un patch tipic per clasă. Observații vizuale relevante pentru proiectare:\n")
    lines.append("- `River` și `Highway` sunt ambele structuri liniare în imagini satelitare → confuzii frecvente confirmate ulterior în matricea de confuzie.")
    lines.append("- `HerbaceousVegetation` și `Pasture` au texturi similare verde-uniform → la fel.")
    lines.append("- `Residential` și `Industrial` au pattern-uri de acoperișuri ușor diferite — diferența e subtilă chiar și pentru un observator uman.")
    lines.append("- `SeaLake` și `Forest` sunt cele mai bine separabile vizual (omogenitate cromatică distinctă).\n")
    lines.append("Exemple concrete de erori sunt în [cnn_misclassified.png](cnn_misclassified.png).\n")

    lines.append("## Pipeline preprocesare + augmentări\n")
    lines.append("**Inferență (test-time):**\n")
    lines.append("```\nResize(224, 224)  ->  ToTensor()  ->  Normalize(ImageNet mean/std)\n```")
    lines.append("Upsamplingul 64->224 este necesar pentru backbone-ul MobileNetV3-Small pretrenat pe ImageNet (224 este input-ul standard).\n")
    lines.append("\n**Augmentări la antrenare (numai pe train, val/test deterministe):**")
    lines.append("- `RandomHorizontalFlip()` — imaginile satelitare nu au orientare stânga/dreapta canonică")
    lines.append("- `RandomVerticalFlip()` — nici nord/sud (Sentinel-2 capturează la orbite multiple)")
    lines.append("- `RandomRotation(15°)` — modelează variațiile de unghi orbital")
    lines.append("- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` — modelează variabilitatea sezonieră și atmosferică\n")
    lines.append("Normalizarea folosește media și deviația standard de pe ImageNet (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) — obligatoriu pentru ca greutățile pretrenate să primească exact distribuția pe care au fost antrenate.\n")

    (OUT_DIR / "cnn_eda.md").write_text("\n".join(lines))
    print(f"  -> {OUT_DIR / 'cnn_eda.md'}")


if __name__ == "__main__":
    main()
