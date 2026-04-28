"""
    Preluam date din catalogul de cutremure USGS (United States Geological Survey)
    Se vor executa mai multe query-uri pentru fiecare 10 zile din 1990 pana in 2026, astfel incat sa nu depaseasca
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
INTERVAL_DAYS = 10

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
SLEEP_SECONDS = 0.4
MAX_RETRIES = 3


def daterange(start: datetime, end: datetime, step_days: int = INTERVAL_DAYS):
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=step_days)


def fetch_interval(start: datetime) -> pd.DataFrame | None:
    end = min(start + timedelta(days=INTERVAL_DAYS), END_DATE)
    params = {
        "format": "csv",
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime": end.strftime("%Y-%m-%d"),
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
        start_from = last_done + timedelta(days=INTERVAL_DAYS)
        print(f"Resuming from {start_from.date()} (last completed: {last_done.date()})")
    else:
        start_from = START_DATE
        print(f"Starting fresh from {start_from.date()}")

    write_header = not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0
    total_intervals = ((END_DATE - start_from).days // INTERVAL_DAYS) + 1
    total_events = 0
    failed_intervals = []

    with OUTPUT_FILE.open("a", encoding="utf-8") as out_fh:
        for i, interval_start in enumerate(daterange(start_from, END_DATE)):
            interval_end = min(interval_start + timedelta(days=INTERVAL_DAYS), END_DATE)

            try:
                df = fetch_interval(interval_start)
            except Exception as e:
                print(f"{interval_start.date()} – {interval_end.date()}: FAILED after retries — {e}")
                failed_intervals.append(interval_start)
                continue

            if df is None or df.empty:
                save_progress(interval_start)
                if i % 10 == 0:
                    print(f"{interval_start.date()} – {interval_end.date()}: 0 events (interval {i+1}/{total_intervals})")
                time.sleep(SLEEP_SECONDS)
                continue

            df.to_csv(out_fh, index=False, header=write_header)
            write_header = False
            total_events += len(df)
            save_progress(interval_start)

            if i % 10 == 0 or len(df) > 5000:
                print(
                    f"{interval_start.date()} – {interval_end.date()}: {len(df):>5,} events "
                    f"(running total {total_events:,}, interval {i+1}/{total_intervals})"
                )

            time.sleep(SLEEP_SECONDS)

    print(f"\nDone. Total events fetched this session: {total_events:,}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
    if failed_intervals:
        print(f"\n{len(failed_intervals)} intervals failed and were skipped:")
        for d in failed_intervals[:20]:
            print(f"  {d.date()} – {(d + timedelta(days=INTERVAL_DAYS)).date()}")
        if len(failed_intervals) > 20:
            print(f"  ... and {len(failed_intervals) - 20} more")

if __name__ == "__main__":
    main()