"""Download and extract EuroSAT (RGB version)."""
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "model2"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DATA_DIR / "EuroSAT.zip"

URLS = [
    "https://huggingface.co/datasets/blanchon/EuroSAT_RGB/resolve/main/EuroSAT_RGB.zip",
    "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
    "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip",
]


def download(url, path):
    print(f"Trying {url} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done / total * 100
                print(f"\r  {done / 1024**2:.0f} / {total / 1024**2:.0f} MB ({pct:.0f}%)", end="")
    print()


def main():
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 100_000_000:
        print(f"Already downloaded: {ZIP_PATH}")
    else:
        for url in URLS:
            try:
                download(url, ZIP_PATH)
                break
            except Exception as e:
                print(f"  Failed: {e}")
        else:
            raise RuntimeError("All download mirrors failed")
        print(f"Saved to {ZIP_PATH}")

    extract_dir = DATA_DIR / "extracted"
    if not extract_dir.exists() or not list(extract_dir.iterdir()):
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")
    else:
        print(f"Already extracted at {extract_dir}")

    candidates = list(extract_dir.glob("**/AnnualCrop"))
    if candidates:
        class_root = candidates[0].parent
        classes = sorted([p.name for p in class_root.iterdir() if p.is_dir()])
        print(f"\nClasses ({len(classes)}) at {class_root}:")
        for c in classes:
            n = len(list((class_root / c).iterdir()))
            print(f"  {c}: {n:,} images")
    else:
        print("Could not locate class folders — check extracted directory manually")


if __name__ == "__main__":
    main()