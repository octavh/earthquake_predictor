"""
    Preluam date din catalogul de cutremure USGS (United States Geological Survey)
"""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import io

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "earthquakes.csv"

START_YEAR = 1990
END_YEAR = datetime.now().year
MIN_MAGNITUDE = 2.5

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def month_range(start_year: int, end_year: int):
    today = datetime.now()
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if y == today.year and m > today.month:
                return
            yield y, m


def next_month(year: int, month: int):
    if month == 12:
        return year + 1, 1
    return year, month + 1


def fetch_month(year: int, month: int) -> pd.DataFrame:
    next_y, next_m = next_month(year, month)
    params = {
        "format": "csv",
        "starttime": f"{year}-{month:02d}-01",
        "endtime": f"{next_y}-{next_m:02d}-01",
        "minmagnitude": MIN_MAGNITUDE,
        "orderby": "time-asc",
    }
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def main():
    chunks = []
    total = 0
    for year, month in month_range(START_YEAR, END_YEAR):
        try:
            df = fetch_month(year, month)
            chunks.append(df)
            total += len(df)
            print(f"{year}-{month:02d}: {len(df):>6,} events  (total {total:,})")
            time.sleep(0.5)
        except Exception as e:
            print(f"{year}-{month:02d}: FAILED — {e}")

    if not chunks:
        print("No data fetched. Aborting.")
        return

    full = pd.concat(chunks, ignore_index=True)
    full.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(full):,} events to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()