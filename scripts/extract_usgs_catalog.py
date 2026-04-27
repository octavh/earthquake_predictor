"""
    Preluam date din catalogul de cutremure USGS (United States Geological Survey)
    Se vor executa mai multe query-uri pentru fiecare zi din 1990 pana in 2026, astfel incat sa nu depaseasca
    limita de 20,000 de cutremure per query setata de USGS
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import io
import os

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "earthquakes.csv"
PROGRESS_FILE = OUTPUT_DIR / "earthquakes.progress.txt"

START_DATE = datetime(1990, 1, 1)
END_DATE = datetime.now()
MIN_MAGNITUDE = 0.0

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
SLEEP_SECONDS = 0.4
MAX_RETRIES = 3


def daterange(start: datetime, end: datetime):
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=1)


def fetch_day(day: datetime) -> pd.DataFrame | None:
    next_day = day + timedelta(days=1)
    params = {
        "format": "csv",
        "starttime": day.strftime("%Y-%m-%d"),
        "endtime": next_day.strftime("%Y-%m-%d"),
        "minmagnitude": MIN_MAGNITUDE,
        "orderby": "time-asc",
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code == 204:
                return None
            r.raise_for_status()
            text = r.text.strip()
            if not text or text.count("\n") < 1:
                return None
            return pd.read_csv(io.StringIO(text))
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def load_progress() -> datetime | None:
    if not PROGRESS_FILE.exists():
        return None
    text = PROGRESS_FILE.read_text().strip()
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d")


def save_progress(day: datetime) -> None:
    PROGRESS_FILE.write_text(day.strftime("%Y-%m-%d"))


def main():
    last_done = load_progress()
    if last_done:
        start_from = last_done + timedelta(days=1)
        print(f"Resuming from {start_from.date()} (last completed: {last_done.date()})")
    else:
        start_from = START_DATE
        print(f"Starting fresh from {start_from.date()}")

    write_header = not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0
    total_days = (END_DATE - start_from).days + 1
    total_events = 0
    failed_days = []

    with OUTPUT_FILE.open("a", encoding="utf-8") as out_fh:
        for i, day in enumerate(daterange(start_from, END_DATE)):
            try:
                df = fetch_day(day)
            except Exception as e:
                print(f"{day.date()}: FAILED after retries — {e}")
                failed_days.append(day)
                continue

            if df is None or df.empty:
                save_progress(day)
                if i % 30 == 0:
                    print(f"{day.date()}: 0 events (day {i+1}/{total_days})")
                time.sleep(SLEEP_SECONDS)
                continue

            df.to_csv(out_fh, index=False, header=write_header)
            write_header = False
            total_events += len(df)
            save_progress(day)

            if i % 30 == 0 or len(df) > 5000:
                print(
                    f"{day.date()}: {len(df):>5,} events "
                    f"(running total {total_events:,}, day {i+1}/{total_days})"
                )

            time.sleep(SLEEP_SECONDS)

    print(f"\nDone. Total events fetched this session: {total_events:,}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
    if failed_days:
        print(f"\n{len(failed_days)} days failed and were skipped:")
        for d in failed_days[:20]:
            print(f"  {d.date()}")
        if len(failed_days) > 20:
            print(f"  ... and {len(failed_days) - 20} more")

if __name__ == "__main__":
    main()