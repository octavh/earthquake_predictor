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
sys.path.insert(0, str(ROOT))
from backend.features import LandUseModel, LandUseClassifier

DATA_ROOT = ROOT / "data" / "model2"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 10
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
IMAGE_SIZE = 224
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_class_root(extract_dir: Path) -> Path:
    candidates = list(extract_dir.glob("**/AnnualCrop"))
    if not candidates:
        raise RuntimeError(f"Could not find class folders under {extract_dir}")
    return candidates[0].parent


def build_transforms():
    train_t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_t, eval_t


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, label = self.subset[i]
        return self.transform(img), label


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total


def train_phase(name, model, train_loader, val_loader, device, optimizer, criterion, epochs, scheduler=None):
    print(f"\n=== Phase: {name} ({epochs} epochs) ===")
    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        if scheduler is not None:
            scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        best_val = max(best_val, val_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch}/{epochs}  "
            f"train_loss={loss_sum/total:.4f}  "
            f"train_acc={correct/total:.3f}  "
            f"val_acc={val_acc:.3f}  "
            f"lr={lr_now:.1e}"
        )
    return best_val


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

    train_t, eval_t = build_transforms()

    base_dataset = datasets.ImageFolder(str(class_root))
    classes = base_dataset.classes
    num_classes = len(classes)
    print(f"Total images: {len(base_dataset):,}, classes: {num_classes}")
    print(f"Classes: {classes}")

    expected = LandUseClassifier.EUROSAT_CLASSES
    assert classes == expected, (
        f"ImageFolder classes {classes} do not match LandUseClassifier.EUROSAT_CLASSES {expected}. "
        f"Vulnerability scoring would be silently wrong. Check the data layout under {class_root}."
    )

    n_test = int(len(base_dataset) * 0.1)
    n_val = int(len(base_dataset) * 0.1)
    n_train = len(base_dataset) - n_val - n_test
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_ds = TransformSubset(train_subset, train_t)
    val_ds = TransformSubset(val_subset, eval_t)
    test_ds = TransformSubset(test_subset, eval_t)
    print(f"Splits: train={n_train}, val={n_val}, test={n_test}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)

    print("Building MobileNetV3-Small with ImageNet pretrained weights...")
    model = LandUseModel(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    print("Freezing backbone, training new classifier head only...")
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.classifier[3].parameters():
        p.requires_grad = True

    head_params = [p for p in model.parameters() if p.requires_grad]
    head_opt = optim.Adam(head_params, lr=HEAD_LR)
    train_phase("head warmup", model, train_loader, val_loader, device, head_opt, criterion, HEAD_EPOCHS)

    print("Unfreezing all parameters for full-network fine-tune...")
    for p in model.parameters():
        p.requires_grad = True

    ft_opt = optim.Adam(model.parameters(), lr=FINETUNE_LR)
    ft_sched = optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=FINETUNE_EPOCHS)
    best_val = train_phase(
        "full fine-tune", model, train_loader, val_loader, device,
        ft_opt, criterion, FINETUNE_EPOCHS, scheduler=ft_sched,
    )

    print("\n=== Test set performance ===")
    model.eval()
    all_preds, all_truth = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            all_preds.extend(out.argmax(1).cpu().numpy().tolist())
            all_truth.extend(yb.numpy().tolist())

    test_acc = float((np.array(all_preds) == np.array(all_truth)).mean())
    print(f"test_acc={test_acc:.4f}  best_val_acc={best_val:.4f}")
    print(classification_report(all_truth, all_preds, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(all_truth, all_preds))

    out_path = MODELS_DIR / "cnn_eurosat.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "input_size": IMAGE_SIZE,
        "model_arch": "mobilenet_v3_small",
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std": IMAGENET_STD,
        "val_acc": best_val,
        "test_acc": test_acc,
    }, out_path)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
