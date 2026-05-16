import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import io
import sys

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "model1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USGS_RAW = OUTPUT_DIR / "earthquakes-raw.csv"
INFP_RAW = OUTPUT_DIR / "vrancea-raw.csv"
FINAL_OUTPUT = OUTPUT_DIR / "earthquakes.csv"
PROGRESS_FILE = OUTPUT_DIR / "earthquakes.extraction_progress.txt"
INFP_MARKER = OUTPUT_DIR / ".infp_downloaded"

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_START = datetime(1990, 1, 1)
USGS_END = datetime.now()
USGS_MIN_MAG = 0.0
USGS_INTERVAL_DAYS = 10
USGS_SLEEP_SECONDS = 0.4
USGS_MAX_RETRIES = 3

INFP_URL = "https://www.infp.ro/data/romplus.txt"
INFP_TIMEOUT = 60

ROMPLUS_COLSPECS = [
    (0, 11), (11, 24), (24, 30), (30, 37), (37, 47), (47, 58),
    (58, 64), (64, 70), (70, 75), (75, 82), (82, 87), (87, 92),
    (92, 97), (97, 102), (102, 107), (107, 112), (112, 117),
]
ROMPLUS_NAMES = [
    'DATE', 'TIME', 'Err', 'RMS', 'LATITUDE', 'LONGITUDE', 'Smaj', 'Smin',
    'Az', 'DEPTH', 'DepthErr', 'ML', 'MLErr', 'MD', 'MDErr', 'Mw', 'MwErr'
]


def daterange(start: datetime, end: datetime, step_days: int = USGS_INTERVAL_DAYS):
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=step_days)


def fetch_usgs_interval(start: datetime) -> pd.DataFrame | None:
    end = min(start + timedelta(days=USGS_INTERVAL_DAYS), USGS_END)
    params = {
        "format": "csv",
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime": end.strftime("%Y-%m-%d"),
        "minmagnitude": USGS_MIN_MAG,
        "orderby": "time-asc",
    }

    for attempt in range(USGS_MAX_RETRIES):
        try:
            r = requests.get(USGS_URL, params=params, timeout=120)
            if r.status_code == 204:
                return None
            r.raise_for_status()
            text = r.text.strip()
            if not text or text.count("\n") < 1:
                return None
            return pd.read_csv(io.StringIO(text))
        except Exception as e:
            if attempt < USGS_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def load_usgs_progress() -> datetime | None:
    if not PROGRESS_FILE.exists():
        return None
    text = PROGRESS_FILE.read_text().strip()
    if not text or text.startswith("INFP"):
        return None
    try:
        return datetime.strptime(text.split("|")[0], "%Y-%m-%d")
    except Exception:
        return None

def save_usgs_progress(day: datetime) -> None:
    PROGRESS_FILE.write_text(day.strftime("%Y-%m-%d"))


def extract_usgs():
    print("\n" + "=" * 70)
    print("FAZA 1: SCARITA DATE USGS")
    print("=" * 70)

    last_done = load_usgs_progress()
    if last_done:
        start_from = last_done + timedelta(days=USGS_INTERVAL_DAYS)
        print(f"\nReiau de la {start_from.date()} (ultima completa: {last_done.date()})")
    else:
        start_from = USGS_START
        print(f"\nIncepe de la {start_from.date()}")

    write_header = not USGS_RAW.exists() or USGS_RAW.stat().st_size == 0
    total_intervals = ((USGS_END - start_from).days // USGS_INTERVAL_DAYS) + 1
    total_events = 0
    failed_intervals = []

    with USGS_RAW.open("a", encoding="utf-8") as out_fh:
        for i, interval_start in enumerate(daterange(start_from, USGS_END)):
            interval_end = min(interval_start + timedelta(days=USGS_INTERVAL_DAYS), USGS_END)

            try:
                df = fetch_usgs_interval(interval_start)
            except Exception as e:
                print(f"{interval_start.date()} – {interval_end.date()}: EROARE — {e}")
                failed_intervals.append(interval_start)
                continue

            if df is None or df.empty:
                save_usgs_progress(interval_start)
                if i % 10 == 0:
                    print(f"{interval_start.date()} – {interval_end.date()}: 0 (interval {i+1}/{total_intervals})")
                time.sleep(USGS_SLEEP_SECONDS)
                continue

            df.to_csv(out_fh, index=False, header=write_header)
            write_header = False
            total_events += len(df)
            save_usgs_progress(interval_start)

            if i % 10 == 0 or len(df) > 5000:
                print(f"{interval_start.date()} – {interval_end.date()}: {len(df):>5,} "
                      f"(total {total_events:,}, interval {i+1}/{total_intervals})")

            time.sleep(USGS_SLEEP_SECONDS)

    print(f"\n✓ USGS scarita completa")
    print(f"  Total: {total_events:,}")
    print(f"  Fisier: {USGS_RAW.name}")
    if failed_intervals:
        print(f"\n⚠️  {len(failed_intervals)} intervale esec:")
        for d in failed_intervals[:5]:
            print(f"  {d.date()} – {(d + timedelta(days=USGS_INTERVAL_DAYS)).date()}")


def extract_infp():
    print("\n" + "=" * 70)
    print("FAZA 2: SCARITA DATE INFP ROMPLUS")
    print("=" * 70)

    if INFP_MARKER.exists():
        print("\n✓ INFP deja scaritata (sare peste)")
        if INFP_RAW.exists():
            lines = INFP_RAW.read_text().count('\n')
            print(f"  Inregistrari: {max(0, lines - 1):,}")
        return

    print(f"\nScarita de la: {INFP_URL}")
    try:
        r = requests.get(INFP_URL, timeout=INFP_TIMEOUT)
        r.raise_for_status()
        text = r.text.strip()

        if not text:
            raise ValueError("Raspuns gol de la INFP")

        INFP_RAW.write_text(text, encoding='utf-8')
        num_records = text.count('\n') - 1

        print(f"\n✓ Scaritata {num_records:,} cutremure istorice INFP")
        print(f"  Fisier: {INFP_RAW.name}")
        INFP_MARKER.touch()

    except requests.exceptions.RequestException as e:
        print(f"\n⚠️  Esec scarita INFP: {e}")
        print("  (USGS singura e suficienta)")


def load_and_filter_usgs() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("FAZA 3: PROCESEAZA DATE USGS")
    print("=" * 70)

    print(f"\nIncarc {USGS_RAW.name}...")
    if not USGS_RAW.exists():
        raise FileNotFoundError(f"Lipseste {USGS_RAW}")

    df = pd.read_csv(USGS_RAW, on_bad_lines='skip')
    total = len(df)
    print(f"  Total inregistrari: {total:,}")

    df = df[df['type'] == 'earthquake'].copy()
    removed = total - len(df)
    print(f"  Eliminat {removed:,} non-naturale (nucleare, dinamita, etc)")

    df = df[['time', 'latitude', 'longitude', 'depth', 'mag']].copy()

    before = len(df)
    df = df[df['mag'] > 0]
    print(f"  Eliminat {before - len(df):,} cu magnitude = 0")

    before = len(df)
    df = df[df['time'].str[:4].astype(int) >= 1990]
    print(f"  Eliminat {before - len(df):,} inainte de 1990")

    print(f"  Final: {len(df):,} cutremure valide")

    return df


def load_and_parse_infp() -> pd.DataFrame:
    print(f"\nIncarc {INFP_RAW.name}...")
    if not INFP_RAW.exists():
        print("  (Fisier INFP lipsit - fara date istorice)")
        return pd.DataFrame(columns=['time', 'latitude', 'longitude', 'depth', 'mag'])

    df = pd.read_fwf(
        INFP_RAW,
        colspecs=ROMPLUS_COLSPECS,
        names=ROMPLUS_NAMES,
        skiprows=1,
        dtype=str
    )
    print(f"  Total inregistrari: {len(df):,}")

    def build_time(row):
        try:
            date_str = str(row['DATE']).strip()
            time_str = str(row['TIME']).strip()
            parts = date_str.split('/')
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            return f'{year:04d}-{month:02d}-{day:02d}T{time_str}Z'
        except Exception:
            return np.nan

    df['time'] = df.apply(build_time, axis=1)

    df['depth'] = pd.to_numeric(df['DEPTH'].str.strip().str.rstrip('f'), errors='coerce')
    df['latitude'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')

    def best_mag(row):
        for col in ['Mw', 'ML', 'MD']:
            try:
                v = float(str(row[col]).strip())
                if v != 0.0 and not np.isnan(v):
                    return v
            except Exception:
                pass
        return np.nan

    df['mag'] = df.apply(best_mag, axis=1)

    df = df[['time', 'latitude', 'longitude', 'depth', 'mag']].dropna(subset=['time'])

    before = len(df)
    df = df[df['mag'] > 0]
    print(f"  Eliminat {before - len(df):,} cu magnitude = 0")

    before = len(df)
    df = df[df['time'].str[:4].astype(int) >= 1990]
    print(f"  Eliminat {before - len(df):,} inainte de 1990")

    df["time_parsed"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    n_bad = df["time_parsed"].isna().sum()
    if n_bad > 0:
        print(f"  ⚠️ {n_bad} INFP randuri cu timp neparsabil — eliminat")
        df = df.dropna(subset=["time_parsed"])
    df = df.drop(columns=["time_parsed"])

    print(f"  Final: {len(df):,} inregistrari valide")

    return df


def merge_and_clean(usgs_df: pd.DataFrame, infp_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("FAZA 4: UNESTE SI CURATA DATE")
    print("=" * 70)

    print(f"\nUneste USGS + INFP...")
    combined = pd.concat([usgs_df, infp_df], ignore_index=True)
    print(f"  Total: {len(combined):,}")

    print(f"\nCurata...")
    print(f"  Valori nule: depth={combined['depth'].isnull().sum():,}, "
          f"mag={combined['mag'].isnull().sum():,}")

    combined = combined.dropna()
    print(f"  Dupa eliminare nule: {len(combined):,}")

    before_dedup = len(combined)
    combined = combined.drop_duplicates()
    print(f"  Dupa eliminare duplicate: {len(combined):,} (eliminat {before_dedup - len(combined):,})")

    combined["_t_round"] = pd.to_datetime(combined["time"], utc=True, errors="coerce").dt.floor("10s")
    combined["_lat_round"] = (combined["latitude"] * 20).round() / 20
    combined["_lon_round"] = (combined["longitude"] * 20).round() / 20
    before_coarse = len(combined)
    combined = combined.drop_duplicates(subset=["_t_round", "_lat_round", "_lon_round"], keep="first")
    combined = combined.drop(columns=["_t_round", "_lat_round", "_lon_round"])
    print(f"  Eliminat {before_coarse - len(combined):,} duplicate cross-catalog")

    combined = combined.sort_values('time').reset_index(drop=True)

    print(f"\nValidare:")
    print(f"  Latitudine:  {combined['latitude'].min():.2f}° pana {combined['latitude'].max():.2f}°")
    print(f"  Longitudine: {combined['longitude'].min():.2f}° pana {combined['longitude'].max():.2f}°")
    print(f"  Adancime:    {combined['depth'].min():.1f} pana {combined['depth'].max():.1f} km")
    print(f"  Magnitudine: {combined['mag'].min():.2f} pana {combined['mag'].max():.2f}")
    print(f"  Perioada:    {combined['time'].min()} pana {combined['time'].max()}")

    return combined


def save_final(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("FAZA 5: SALVEAZA DATASET FINAL")
    print("=" * 70)

    print(f"\nSalveaza {FINAL_OUTPUT.name}...")
    df.to_csv(FINAL_OUTPUT, index=False)
    file_size_mb = FINAL_OUTPUT.stat().st_size / (1024**2)
    print(f"  ✓ Salvat {len(df):,} inregistrari ({file_size_mb:.1f} MB)")


def main():
    print("=" * 70)
    print("PIPELINE CUTREMURE: SCARITA, EXTRAGE, CURATA, PREGATESTE")
    print("=" * 70)
    print("\nScarita de la USGS si INFP, filtreaza naturale,")
    print("curata, pregateste pentru antrenare model.")

    try:
        extract_usgs()
        extract_infp()

        usgs_df = load_and_filter_usgs()
        infp_df = load_and_parse_infp()

        final_df = merge_and_clean(usgs_df, infp_df)

        save_final(final_df)

        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        if INFP_MARKER.exists():
            INFP_MARKER.unlink()

        print("\n" + "=" * 70)
        print(f"✓ GATA: Pipeline complet!")
        print("=" * 70)
        print(f"\n✓ {len(final_df):,} inregistrari in {FINAL_OUTPUT.name}")

    except Exception as e:
        print(f"\n✗ EROARE: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
