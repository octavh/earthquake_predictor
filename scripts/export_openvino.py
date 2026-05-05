"""
Convert trained models to OpenVINO IR format.

Uses subprocess isolation to avoid library conflicts that segfault when
LightGBM, PyTorch, ONNX, and OpenVINO are all loaded into one Python process.

Note: LightGBM models are exported to ONNX but not converted to OpenVINO IR.
OpenVINO does not support the ai.onnx.ml.TreeEnsembleClassifier operator that
LightGBM uses, so tree-based models stay as .pkl files served via joblib.
The CNN is the model that actually benefits from OpenVINO acceleration.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
THRESHOLDS = [3, 4, 5, 6, 7]
PYTHON = sys.executable


def run_inline(script: str, label: str) -> bool:
    """Run a Python snippet in a fresh subprocess. Returns True on success."""
    print(f"  Converting {label}...")
    result = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"    ✓ {result.stdout.strip()}")
        return True
    err = (result.stderr or "").strip().splitlines()
    last_lines = "\n      ".join(err[-3:]) if err else "(no error message)"
    print(f"    ✗ FAILED: {last_lines}")
    return False


# ---------- LightGBM .pkl -> ONNX (kept as bonus, not converted further) ----------
LGBM_TO_ONNX = """
import joblib
from pathlib import Path
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

src = Path('{src}')
dst = Path('{dst}')
bundle = joblib.load(src)
n = len(bundle['feature_cols'])
onnx_model = convert_lightgbm(
    bundle['model'].booster_,
    initial_types=[('input', FloatTensorType([None, n]))],
    zipmap=False,
)
with dst.open('wb') as f:
    f.write(onnx_model.SerializeToString())
print(f'wrote {{dst.name}}')
"""

# ---------- CNN .pth -> ONNX ----------
CNN_TO_ONNX = """
import sys, torch
from pathlib import Path
sys.path.insert(0, '{root}')
from backend.features import _SmallCNN

bundle = torch.load('{src}', map_location='cpu', weights_only=False)
classes = bundle['classes']
input_size = bundle.get('input_size', 64)

model = _SmallCNN(num_classes=len(classes))
model.load_state_dict(bundle['state_dict'])
model.eval()

dummy = torch.randn(1, 3, input_size, input_size)
torch.onnx.export(
    model, dummy, '{dst}',
    input_names=['input'],
    output_names=['logits'],
    dynamic_axes={{'input': {{0: 'batch'}}, 'logits': {{0: 'batch'}}}},
    opset_version=14,
)
out_name = Path('{dst}').name
print(f'wrote {{out_name}}')
"""

# ---------- ONNX -> OpenVINO IR ----------
ONNX_TO_IR = """
import openvino as ov
from pathlib import Path

src = Path('{src}')
dst = src.with_suffix('.xml')
core = ov.Core()
ov_model = core.read_model(str(src))
ov.save_model(ov_model, str(dst))
print(f'wrote {{dst.name}} (+ .bin)')
"""


def main():
    print("=" * 60)
    print("OpenVINO conversion pipeline")
    print("=" * 60)

    print("\n[1/3] LightGBM .pkl -> .onnx (kept for reference)")
    onnx_files = []
    for m in THRESHOLDS:
        src = MODELS_DIR / f"lgbm_m{m}.pkl"
        dst = MODELS_DIR / f"lgbm_m{m}.onnx"
        if not src.exists():
            print(f"  skipping {src.name} (not found)")
            continue
        script = LGBM_TO_ONNX.format(src=src, dst=dst)
        if run_inline(script, f"lgbm_m{m}"):
            onnx_files.append(dst)

    print("\n[2/3] CNN .pth -> .onnx")
    cnn_src = MODELS_DIR / "cnn_eurosat.pth"
    cnn_dst = MODELS_DIR / "cnn_eurosat.onnx"
    if cnn_src.exists():
        script = CNN_TO_ONNX.format(root=ROOT, src=cnn_src, dst=cnn_dst)
        if run_inline(script, "cnn_eurosat"):
            onnx_files.append(cnn_dst)
    else:
        print(f"  skipping {cnn_src.name} (not found)")

    print("\n[3/3] .onnx -> OpenVINO IR (.xml + .bin)")
    for onnx_path in onnx_files:
        # Skip LightGBM models — OpenVINO doesn't support tree-ensemble operators
        if onnx_path.stem.startswith("lgbm_"):
            print(f"  skipping {onnx_path.name} (LightGBM trees not supported by OpenVINO)")
            continue
        if not onnx_path.exists():
            print(f"  skipping {onnx_path.name} (not produced)")
            continue
        script = ONNX_TO_IR.format(src=onnx_path)
        run_inline(script, onnx_path.stem)

    print("\n" + "=" * 60)
    print("Final state of models/ directory:")
    print("=" * 60)
    for f in sorted(MODELS_DIR.iterdir()):
        if f.suffix in (".pkl", ".pth", ".onnx", ".xml", ".bin"):
            kb = f.stat().st_size / 1024
            size = f"{kb:.1f} KB" if kb < 1024 else f"{kb/1024:.1f} MB"
            print(f"  {f.name:35s} {size:>12s}")

    have_ir = list(MODELS_DIR.glob("*.xml"))
    if have_ir:
        print(f"\n✓ Produced {len(have_ir)} OpenVINO IR file(s):")
        for f in have_ir:
            print(f"    - {f.name}")
        print("\nThese satisfy the competition's OpenVINO requirement.")
    else:
        print("\n✗ No OpenVINO IR files produced. Check error messages above.")


if __name__ == "__main__":
    main()