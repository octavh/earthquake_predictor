"""Convert trained models to OpenVINO IR format for optimized inference."""
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT))

THRESHOLDS = [3, 4, 5, 6, 7]


class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 4 * 4, 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def export_lightgbm_to_onnx(threshold: int):
    """Convert one LightGBM model to ONNX -> OpenVINO IR."""
    from onnxmltools import convert_lightgbm
    from onnxconverter_common.data_types import FloatTensorType

    src = MODELS_DIR / f"lgbm_m{threshold}.pkl"
    if not src.exists():
        print(f"  Missing {src.name} — skipping")
        return

    bundle = joblib.load(src)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    n_features = len(feature_cols)

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(model.booster_, initial_types=initial_type, zipmap=False)

    onnx_path = MODELS_DIR / f"lgbm_m{threshold}.onnx"
    with onnx_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Saved {onnx_path.name}")
    return onnx_path


def export_cnn_to_onnx():
    """Convert the CNN to ONNX."""
    src = MODELS_DIR / "cnn_eurosat.pth"
    if not src.exists():
        print(f"  Missing {src.name} — skipping")
        return

    bundle = torch.load(src, map_location="cpu", weights_only=False)
    classes = bundle["classes"]
    input_size = bundle.get("input_size", 64)

    model = SmallCNN(num_classes=len(classes))
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    onnx_path = MODELS_DIR / "cnn_eurosat.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    print(f"  Saved {onnx_path.name}")
    return onnx_path


def convert_to_openvino_ir(onnx_path: Path):
    """Convert an ONNX file to OpenVINO IR format."""
    try:
        import openvino as ov
        print(f"  Converting {onnx_path.name} to OpenVINO IR...")
        core = ov.Core()
        ov_model = core.read_model(str(onnx_path))
        out_path = onnx_path.with_suffix(".xml")
        ov.save_model(ov_model, str(out_path))
        print(f"  Saved {out_path.name} (+ .bin)")
        return out_path
    except Exception as e:
        print(f"  WARNING: OpenVINO conversion failed: {e}")
        print(f"  Keeping ONNX format: {onnx_path.name}")
        return None


def main():
    print("=== Exporting LightGBM models to ONNX ===")
    onnx_paths = []
    for m in THRESHOLDS:
        try:
            onnx_path = export_lightgbm_to_onnx(m)
            if onnx_path:
                onnx_paths.append(onnx_path)
        except Exception as e:
            print(f"  M>={m} FAILED: {e}")

    print("\n=== Exporting CNN to ONNX ===")
    try:
        onnx_path = export_cnn_to_onnx()
        if onnx_path:
            onnx_paths.append(onnx_path)
    except Exception as e:
        print(f"  CNN FAILED: {e}")

    print("\n=== Converting to OpenVINO IR (optional) ===")
    for onnx_path in onnx_paths:
        try:
            convert_to_openvino_ir(onnx_path)
        except Exception as e:
            print(f"  OpenVINO conversion for {onnx_path.name} failed: {e}")

    print("\n✓ Done. Optimized models in models/:")
    for f in sorted(MODELS_DIR.iterdir()):
        if f.suffix in (".onnx", ".xml", ".bin"):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()