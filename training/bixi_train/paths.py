"""Shared configuration for the offline training pipeline.

Everything here is OFFLINE ONLY. Python cannot run in a Worker, so this package
never ships; it produces one artifact JSON that src/gbdt.ts walks at inference.

The constants that also exist on the TypeScript side (bucket edges, the morning
window, the feature order) are duplicated here under protest. That duplication is
exactly the train/serve skew risk the design worries about, so it is contained
two ways: the edges are written INTO the artifact by export.py and read back by
src/features.ts (rather than being hardcoded there), and scripts/parity.ts proves
the two encoders agree on ~500 raw rows before any artifact may activate.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"  # downloads + intermediates, gitignored, all rebuildable
OUT = ROOT / "out"  # artifacts + reports, gitignored

TZ = ZoneInfo("America/Toronto")
TZ_NAME = "America/Toronto"

# Station 345 in bixi-monitor. Open data has NO station id column, so the join
# key across the whole pipeline is this exact name string.
STATION_345 = "Regina / de Verdun"
STATION_345_CAPACITY = 19

# Yearly trip archives. 2026 exists but covers Jan-Apr only (open data lags about
# a quarter), so it contains no summer and is not used for training.
ZIP_URLS = {
    2024: "https://cdn.bixi.com/wp-content/uploads/2025/01/DonneesOuvertes2024_010203040506070809101112.zip",
    2025: "https://cdn.bixi.com/wp-content/uploads/2026/02/DonneesOuvertes2025_010203040506070809101112.zip",
}
TRAIN_YEARS = (2024, 2025)

# The window everything is normalised against: 06:00-10:59 local. The 402k-row
# day-level measurement used this window, demand.ts calibrates the nightly level
# over it, and the offline station level below is defined by it. All three have
# to mean the same thing or the level calibration silently rescales the model.
MORNING_HOURS = (6, 7, 8, 9, 10)

# Hours the panel keeps. The simulation starts at 22:00 and runs to noon, so
# these are the only hours whose demand is ever consumed. Dropping 13:00-21:00
# removes ~38% of rows that no prediction reads.
PANEL_HOURS = tuple(list(range(0, 13)) + [22, 23])

# Cycling season. Outside this BIXI is closed or barely running, and off-season
# rows would teach the model a winter shape it is never asked to predict.
SEASON_START_MD = (4, 15)
SEASON_END_MD = (11, 15)

# Bucket edges, lifted from bixi-predictor's PARAMS so both services agree what
# "a wet morning" is. They are shipped inside the artifact; TypeScript reads them
# from there rather than redeclaring them.
PRECIP_DRY_MAX_MM = 0.5
PRECIP_WET_MIN_MM = 4.0
PROB_BUMP_THRESHOLD = 50.0  # Open-Meteo precipitation_probability is 0-100

# Weather: one archive call covers both seasons. Kept a day either side of the
# season so the night-precipitation window (previous 21:00-23:00) is never short.
WEATHER_START = "2023-12-31"
WEATHER_END = "2026-01-02"
WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=45.4673&longitude=-73.5708"
    "&start_date={start}&end_date={end}"
    "&hourly=temperature_2m,precipitation,wind_speed_10m,dew_point_2m"
    "&timezone=America%2FToronto"
)


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)


def dst_intervals(years: tuple[int, ...]) -> list[tuple[int, int]]:
    """UTC epoch ranges during which Montreal is on EDT (UTC-4).

    The trip CSVs carry epoch milliseconds UTC and the panel needs local hours.
    A blanket -14400 is exact from March to early November, which is why the
    day-level exploration could get away with it -- but the season runs to
    November 15, and those last ~12 days are EST. Assuming otherwise would shift
    every hour label by one across the shoulder of the season, smearing the
    morning peak precisely where the model is most sensitive.

    Derived from the tz database rather than from the US transition rule, so it
    stays correct if the rule ever changes.
    """
    out: list[tuple[int, int]] = []
    for y in years:
        start = end = None
        t = dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc)
        prev = t.astimezone(TZ).utcoffset()
        for _ in range(366 * 24):
            t += dt.timedelta(hours=1)
            cur = t.astimezone(TZ).utcoffset()
            if cur != prev:
                if cur == dt.timedelta(hours=-4):
                    start = int(t.timestamp())
                else:
                    end = int(t.timestamp())
                prev = cur
            if t.year > y:
                break
        if start is None or end is None:
            raise RuntimeError(f"no DST transition found for {y}")
        out.append((start, end))
    return out
