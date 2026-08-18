from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, DATA_URLS, RAW_CSV

COLUMNS = [
    "datetime",
    "city",
    "state",
    "country",
    "shape",
    "duration_seconds",
    "duration_hours_min",
    "comments",
    "date_posted",
    "latitude",
    "longitude",
]

TIDY_RENAME = {
    "date_time": "datetime",
    "city_area": "city",
    "ufo_shape": "shape",
    "encounter_length": "duration_seconds",
    "described_encounter_length": "duration_hours_min",
    "description": "comments",
    "date_documented": "date_posted",
}


def ensure_data() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and RAW_CSV.stat().st_size > 0:
        return RAW_CSV

    last_error: Exception | None = None
    for url in DATA_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                RAW_CSV.write_bytes(response.read())
            return RAW_CSV
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Impossible de telecharger les donnees: {last_error}")


def _has_header(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        first_row = next(csv.reader(handle))
    lowered = [cell.strip().lower() for cell in first_row]
    return "datetime" in lowered or "date_time" in lowered


def load_reports(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or ensure_data()
    if _has_header(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.rename(columns=TIDY_RENAME)
        missing = [column for column in COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Colonnes absentes: {missing}")
        df = df[COLUMNS]
    else:
        df = pd.read_csv(csv_path, names=COLUMNS, low_memory=False)

    df["observed_at"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["posted_at"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df["comments"] = df["comments"].fillna("").astype(str)
    df["shape"] = df["shape"].fillna("").astype(str).str.strip().str.lower()
    return df
