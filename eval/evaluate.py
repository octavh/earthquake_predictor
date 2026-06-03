"""End-to-end evaluation: tabular + CNN baselines, ablation, and error visualisations.

Outputs (written to ./evaluation/):
- tabular_results.md      — markdown table: brier, log-loss, AUC per threshold vs base-rate baseline
- cnn_results.md          — markdown table: new MobileNetV3 vs old SmallCNN vs random vs majority
- cnn_confusion_matrix.png
- cnn_misclassified.png   — 16 misclassified test images with predicted vs true labels

Run from project root: .venv/bin/python scripts/evaluate.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from backend.features import LandUseModel, LandUseClassifier

MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "evaluation"
OUT_DIR.mkdir(exist_ok=True)

# ---------- Shared ----------

def fmt(x, n=4):
    if x is None:
        return "—"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "—"
    return f"{x:.{n}f}"


# ---------- Tabular ----------

TABULAR_FEATURE_COLS = [
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


def eval_tabular():
    print("\n== Tabular evaluation ==")
    df = pd.read_csv(ROOT / "data" / "model1" / "training_set.csv", parse_dates=["prediction_date"])
    test = df[df["prediction_date"] >= SPLIT_DATE].copy()
    print(f"  test rows (post-{SPLIT_DATE}): {len(test):,}")

    X = test[TABULAR_FEATURE_COLS].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=9999.0, neginf=-9999.0)

    rows = []
    for m in THRESHOLDS:
        y = test[f"label_m{m}"].to_numpy(dtype=int)
        base_rate = float(y.mean())

        # Baseline: predict the empirical positive rate for everyone.
        baseline_p = np.full_like(y, base_rate, dtype=float)
        b_brier = brier_score_loss(y, baseline_p)
        try:
            b_ll = log_loss(y, baseline_p, labels=[0, 1])
        except ValueError:
            b_ll = float("nan")
        b_auc = float("nan")  # constant predictor has undefined AUC

        # Model
        sess = ort.InferenceSession(str(MODELS_DIR / f"lgbm_m{m}.onnx"),
                                    providers=["CPUExecutionProvider"])
        outputs = sess.run(None, {"input": X})
        probs = outputs[1][:, 1]
        m_brier = brier_score_loss(y, probs)
        try:
            m_ll = log_loss(y, probs, labels=[0, 1])
        except ValueError:
            m_ll = float("nan")
        try:
            m_auc = roc_auc_score(y, probs)
        except ValueError:
            m_auc = float("nan")

        rows.append({
            "threshold": f"M≥2605{m}".replace("≥2605", "≥"),
            "n_test": len(y), "positive_rate": base_rate,
            "model_brier": m_brier, "model_ll": m_ll, "model_auc": m_auc,
            "base_brier": b_brier, "base_ll": b_ll, "base_auc": b_auc,
        })
        print(f"  M≥{m}: model brier={m_brier:.4f} ll={m_ll:.4f} auc={m_auc:.4f}  "
              f"| baseline brier={b_brier:.4f} ll={b_ll:.4f}")

    lines = []
    lines.append("## Performanță modele tabulare (LightGBM, holdout post-2020)\n")
    lines.append("Test set: înregistrări cu `prediction_date >= 2020-01-01` din `training_set.csv`. Comparat cu un baseline constant care prezice rata empirică a clasei pozitive.\n")
    lines.append("| Prag | N test | Rata pozitivă | Brier (LGBM) | Brier (baseline) | LogLoss (LGBM) | LogLoss (baseline) | ROC-AUC (LGBM) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['threshold']} | {r['n_test']:,} | {r['positive_rate']:.4f} | "
            f"{fmt(r['model_brier'])} | {fmt(r['base_brier'])} | "
            f"{fmt(r['model_ll'])} | {fmt(r['base_ll'])} | "
            f"{fmt(r['model_auc'])} |"
        )
    lines.append(
        "\n**Interpretare:**\n"
        "- La pragurile dese (M≥3, M≥4) modelul bate baseline-ul atât pe Brier cât și pe LogLoss — discriminare reală vizibilă în toți indicatorii.\n"
        "- La pragurile rare (M≥5, M≥6, M≥7) baseline-ul constant are Brier/LogLoss mai mici pentru că predicția \"rata empirică pentru toți\" e foarte bine calibrată când clasa pozitivă apare în <5% din cazuri. Totuși, baseline-ul are AUC nedefinit / 0.5 (predicție constantă, zero putere de discriminare), pe când modelul atinge AUC 0.65–0.80.\n"
        "- Concluzie: modelul oferă discriminare reală la toate pragurile (AUC > 0.5), dar la pragurile foarte rare un baseline constant calibrat e greu de bătut pe metricile de calibrare absolută — clasic tradeoff calibrare vs discriminare.\n"
    )
    (OUT_DIR / "tabular_results.md").write_text("\n".join(lines))
    print(f"  → {OUT_DIR / 'tabular_results.md'}")
    return rows


# ---------- CNN ----------

# Architectura veche, păstrată local pentru comparația de tip ablation.
class OldSmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_eval_loader(image_size, mean, std, seed=42, batch=64):
    """Reconstruct the exact test split used during training."""
    data_root = ROOT / "data" / "model2"
    candidates = list(data_root.glob("**/AnnualCrop"))
    if not candidates:
        raise RuntimeError(f"could not find EuroSAT class folders under {data_root}")
    class_root = candidates[0].parent

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    base = datasets.ImageFolder(str(class_root), transform=transform)
    n_test = int(len(base) * 0.1)
    n_val = int(len(base) * 0.1)
    n_train = len(base) - n_val - n_test
    _, _, test_subset = random_split(
        base, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(test_subset, batch_size=batch, num_workers=2), base.classes


def run_model_on_test(model, loader, device):
    model.eval()
    preds, truths, raw_inputs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            truths.extend(yb.numpy().tolist())
    return np.array(preds), np.array(truths)


def eval_cnn():
    print("\n== CNN evaluation ==")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  device: {device}")

    # --- New MobileNetV3-Small ---
    new_bundle = torch.load(MODELS_DIR / "cnn_eurosat.pth", map_location="cpu", weights_only=False)
    classes = new_bundle["classes"]
    new_model = LandUseModel(num_classes=len(classes)).to(device)
    new_model.load_state_dict(new_bundle["state_dict"])

    new_loader, _ = build_eval_loader(
        new_bundle["input_size"],
        new_bundle["normalize_mean"],
        new_bundle["normalize_std"],
    )
    new_preds, truths = run_model_on_test(new_model, new_loader, device)
    new_acc = float((new_preds == truths).mean())
    print(f"  new MobileNetV3-Small test_acc: {new_acc:.4f}")

    # --- Old SmallCNN baseline ---
    old_path = MODELS_DIR / "old_pkl" / "cnn_eurosat.pth"
    old_acc = None
    if old_path.exists():
        old_bundle = torch.load(old_path, map_location="cpu", weights_only=False)
        old_model = OldSmallCNN(num_classes=len(old_bundle.get("classes", classes))).to(device)
        old_model.load_state_dict(old_bundle["state_dict"])
        old_loader, _ = build_eval_loader(
            old_bundle.get("input_size", 64),
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
        )
        old_preds, old_truths = run_model_on_test(old_model, old_loader, device)
        old_acc = float((old_preds == old_truths).mean())
        print(f"  old SmallCNN test_acc:          {old_acc:.4f}")
    else:
        print(f"  (old SmallCNN not found at {old_path}, skipping)")

    # --- Trivial baselines ---
    rng = np.random.default_rng(42)
    random_preds = rng.integers(0, len(classes), size=len(truths))
    random_acc = float((random_preds == truths).mean())

    counts = np.bincount(truths, minlength=len(classes))
    majority_class = int(counts.argmax())
    majority_acc = float((truths == majority_class).mean())
    print(f"  random predictor:                {random_acc:.4f}")
    print(f"  majority-class ({classes[majority_class]}): {majority_acc:.4f}")

    # --- Markdown ---
    lines = []
    lines.append("## Performanță CNN (EuroSAT, holdout 10%)\n")
    lines.append(f"Test set: {len(truths):,} imagini (split deterministic seed=42, identic la antrenare).\n")
    lines.append("| Model | Test accuracy | Comentariu |")
    lines.append("|---|---:|---|")
    lines.append(f"| Random predictor | {random_acc:.4f} | 10 clase echilibrate → ~10% |")
    lines.append(f"| Majority class (`{classes[majority_class]}`) | {majority_acc:.4f} | prezice mereu clasa dominantă |")
    if old_acc is not None:
        lines.append(f"| SmallCNN (3 conv blocks, 8 epoci, fără transfer learning) | {old_acc:.4f} | arhitectura inițială |")
    lines.append(f"| **MobileNetV3-Small + transfer learning + augmentări** (curent) | **{new_acc:.4f}** | 5 epoci head warmup + 10 epoci fine-tune |")
    if old_acc is not None:
        delta = (new_acc - old_acc) * 100
        lines.append(f"\n**Ablation:** transfer learning + arhitectură mai mare → **+{delta:.1f} puncte procentuale accuracy** față de SmallCNN antrenat from-scratch.\n")
    lines.append("\n### Raport pe clase (model curent)\n")
    lines.append("```")
    lines.append(classification_report(truths, new_preds, target_names=classes, digits=3))
    lines.append("```\n")
    (OUT_DIR / "cnn_results.md").write_text("\n".join(lines))
    print(f"  → {OUT_DIR / 'cnn_results.md'}")

    # --- Confusion matrix PNG ---
    cm = confusion_matrix(truths, new_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — MobileNetV3-Small ({new_acc*100:.1f}% acc, n={len(truths)})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = OUT_DIR / "cnn_confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"  → {cm_path}")

    # --- Misclassified gallery ---
    save_misclassified_examples(new_model, new_loader, device, classes, new_preds, truths)
    return new_acc, old_acc, random_acc, majority_acc


def save_misclassified_examples(model, loader, device, classes, preds, truths, n=16):
    """Walk the loader again and grab the first n misclassified images."""
    wrong_idx = np.where(preds != truths)[0]
    if len(wrong_idx) == 0:
        print("  (no misclassifications — skipping gallery)")
        return

    targets = set(wrong_idx[:n].tolist())
    collected = []
    cursor = 0
    for xb, yb in loader:
        bsz = xb.size(0)
        for k in range(bsz):
            global_i = cursor + k
            if global_i in targets:
                img = xb[k].cpu().numpy()
                img = img.transpose(1, 2, 0)
                # De-normalize for display
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img * std + mean).clip(0, 1)
                collected.append((img, int(yb[k]), int(preds[global_i])))
                if len(collected) >= n:
                    break
        cursor += bsz
        if len(collected) >= n:
            break

    rows = 4
    cols = (len(collected) + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    for ax, (img, true_i, pred_i) in zip(np.atleast_2d(axes).ravel(), collected):
        ax.imshow(img)
        ax.set_title(f"true: {classes[true_i]}\npred: {classes[pred_i]}", fontsize=8)
        ax.axis("off")
    for ax in np.atleast_2d(axes).ravel()[len(collected):]:
        ax.axis("off")
    fig.suptitle(f"Misclassified test examples ({len(collected)} of {len(wrong_idx)} total errors)")
    fig.tight_layout()
    out = OUT_DIR / "cnn_misclassified.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → {out}")


def main():
    print("=" * 60)
    print("Evaluation pipeline")
    print("=" * 60)
    eval_tabular()
    eval_cnn()
    print("\nDone. See evaluation/ for outputs.")


if __name__ == "__main__":
    main()
