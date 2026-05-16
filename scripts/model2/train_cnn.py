import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = ROOT / "data" / "model2"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3
SEED = 42


def find_class_root(extract_dir: Path) -> Path:
    candidates = list(extract_dir.glob("**/AnnualCrop"))
    if not candidates:
        raise RuntimeError(f"Could not find class folders under {extract_dir}")
    return candidates[0].parent


class SmallCNN(nn.Module):
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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    class_root = find_class_root(DATA_ROOT)
    print(f"Loading dataset from {class_root}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    full_dataset = datasets.ImageFolder(str(class_root), transform=transform)
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"Total images: {len(full_dataset):,}, classes: {num_classes}")
    print(f"Classes: {classes}")

    n_test = int(len(full_dataset) * 0.1)
    n_val = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Splits: train={n_train}, val={n_val}, test={n_test}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)

    model = SmallCNN(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (out.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss/train_total:.4f} "
            f"train_acc={train_correct/train_total:.3f} "
            f"val_acc={val_correct/val_total:.3f}"
        )

    model.eval()
    all_preds, all_truth = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            all_preds.extend(out.argmax(1).cpu().numpy().tolist())
            all_truth.extend(yb.numpy().tolist())

    print("\n=== Test set performance ===")
    print(classification_report(all_truth, all_preds, target_names=classes))
    print("\nConfusion matrix:")
    print(confusion_matrix(all_truth, all_preds))

    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "input_size": 64,
    }, MODELS_DIR / "cnn_eurosat.pth")
    print(f"\nSaved {MODELS_DIR / 'cnn_eurosat.pth'}")


if __name__ == "__main__":
    main()