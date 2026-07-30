"""Step 2: hourly archive weather for Montreal.

One Open-Meteo archive call covers both seasons -- roughly 17.5k hourly rows,
under a megabyte, free, no key. It is cached on disk because every later step
re-reads it and the API should not be hit again for bytes that cannot change.

Two things this file is careful about:

  * Open-Meteo reports hourly precipitation as the sum over the PRECEDING hour,
    so the row labelled 07:00 covers 06:00-07:00. src/features.ts documents the
    same convention ("label H covers [H-1, H)"), so the two agree and no shift is
    applied anywhere. Getting this backwards would offset rain from the traffic
    it suppresses by an hour.
  * Asking for local time means DST days do not have 24 hours. Spring-forward has
    23 and fall-back has 25, with 01:00 appearing twice. Both are handled
    explicitly below rather than left to whichever row happens to land last in a
    dict.

What is NOT here: precipitation_probability. The archive API has no such field,
so training cannot see one. It arrives only in the forecast, and src/features.ts
applies it as a deterministic bucket promotion outside the model -- which is the
whole reason the encoder can stay identical between training and serving.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict

import pandas as pd

from . import paths

RAW = paths.DATA / "weather_raw.json"
CSV = paths.DATA / "weather_hourly.csv"


def fetch(force: bool = False) -> dict:
    if RAW.exists() and not force:
        return json.loads(RAW.read_text())
    url = paths.WEATHER_URL.format(start=paths.WEATHER_START, end=paths.WEATHER_END)
    print(f"[weather] GET {url}", flush=True)
    with urllib.request.urlopen(url) as r:
        body = r.read()
    RAW.write_bytes(body)
    return json.loads(body)


def build(force: bool = False) -> pd.DataFrame:
    if CSV.exists() and not force:
        return pd.read_csv(CSV)

    js = fetch(force=force)
    h = js["hourly"]
    times = h["time"]

    # Group by (date, hour) first: on the fall-back night two different UTC hours
    # share the label 01:00. Averaging them is the honest reading of "what was
    # the weather at 1am", and it keeps the panel's (date, hour) key unique.
    acc: dict[tuple[str, int], list[list[float]]] = defaultdict(list)
    cols = ("temperature_2m", "precipitation", "wind_speed_10m", "dew_point_2m")
    for i, t in enumerate(times):
        key = (t[:10], int(t[11:13]))
        acc[key].append([h[c][i] for c in cols])

    dupes = sum(1 for v in acc.values() if len(v) > 1)
    rows = []
    for (date, hour), vals in acc.items():
        n = len(vals)
        avg = [sum(v[j] for v in vals) / n if all(v[j] is not None for v in vals) else None for j in range(4)]
        rows.append((date, hour, *avg))

    df = pd.DataFrame(rows, columns=["date", "hour", "temp_c", "precip_mm", "wind_kmh", "dew_c"])
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)

    missing = int(df[["temp_c", "precip_mm", "wind_kmh", "dew_c"]].isna().sum().sum())
    print(
        f"[weather] {len(df)} hourly rows, {df.date.min()}..{df.date.max()}, "
        f"{dupes} duplicated DST hours averaged, {missing} missing values"
    )
    # A gap here would silently become a NaN feature for a whole day, so it is
    # worth knowing about at build time rather than discovering it in a tree.
    if missing:
        print("[weather] WARNING: archive returned nulls -- check before training", file=sys.stderr)

    df.to_csv(CSV, index=False)
    return df


def ensure_table(con) -> None:
    """Materialise the hourly weather into the panel database.

    Every downstream step joins against it on (date, hour), so it lives beside
    the panel rather than being re-read from CSV in three places.
    """
    if not CSV.exists():
        build()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE weather_hourly AS
          SELECT date::DATE AS date, hour::TINYINT AS hour,
                 temp_c::DOUBLE AS temp_c, precip_mm::DOUBLE AS precip_mm,
                 wind_kmh::DOUBLE AS wind_kmh, dew_c::DOUBLE AS dew_c
            FROM read_csv('{CSV}', header=true);
        """
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    paths.ensure_dirs()
    build(force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
