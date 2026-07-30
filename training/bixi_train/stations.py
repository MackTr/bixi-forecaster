"""Per-station profiles: the three scalars that let a network-wide model serve
one station.

Station 345 contributes ~400 of the panel's 5.2M rows. Training on those alone
would be the "ML on 30 rows fits noise" problem the sibling repo correctly warns
about, so the model trains network-wide -- which only works if it is told what
KIND of station each row belongs to. These are those descriptors, and the same
three numbers ride in the artifact so the Worker can describe station 345 to the
model at inference time.

  morning_share  what fraction of the station's daily departures fall in 6-11am.
                 Separates commuter origins from all-day leisure racks.
  log_level      log mean daily 6-11am departures. Doubles as the Poisson OFFSET
                 during training (see train.py), which is what turns the model's
                 output into a dimensionless demand shape.
  net_balance    morning arrivals vs departures, normalised to [-1, 1].

net_balance is measured over the MORNING WINDOW, not the whole day, and that is
not a detail. Over a full season nearly every station returns roughly what it
sends -- bikes come back, and trucks close the remainder -- so a whole-day net
balance is approximately zero everywhere and carries no signal at all. Restricted
to the morning it separates residential origins (345 empties) from downtown
destinations (they fill), which is precisely the distinction the arrivals model
needs, since arrivals are normalised by the DEPARTURE level.

`upto` exists for honest cross-validation: when a fold is scored on later months,
its profiles must be built from earlier ones only. The final artifact uses
everything.
"""

from __future__ import annotations

import argparse
import math
import sys

import duckdb
import pandas as pd

from . import paths
from .panel import connect

MORNING = ",".join(str(h) for h in paths.MORNING_HOURS)


def compute(con: duckdb.DuckDBPyConnection, upto: str | None = None) -> pd.DataFrame:
    """One row per active station: name, morning_share, log_level, net_balance."""
    cutoff = f"AND sd.date < DATE '{upto}'" if upto else ""
    df = con.execute(
        f"""
        WITH sd AS (
          SELECT sd.name, sd.day, sd.date
            FROM station_days sd
            JOIN active_stations a ON a.name = sd.name
           WHERE 1 = 1 {cutoff}
        ), agg AS (
          SELECT sd.name,
                 COUNT(DISTINCT sd.day) AS days,
                 -- from `events`, which still holds all 24 hours; `panel` keeps
                 -- only the hours the simulation reads, so a day total cannot be
                 -- computed there.
                 SUM(COALESCE(e.dep, 0)) AS all_dep,
                 SUM(CASE WHEN e.hour IN ({MORNING}) THEN COALESCE(e.dep, 0) ELSE 0 END) AS morn_dep,
                 SUM(CASE WHEN e.hour IN ({MORNING}) THEN COALESCE(e.arr, 0) ELSE 0 END) AS morn_arr
            FROM sd
            LEFT JOIN events e ON e.name = sd.name AND e.day = sd.day
           GROUP BY 1
        )
        SELECT name, days,
               all_dep, morn_dep, morn_arr,
               CASE WHEN all_dep > 0 THEN morn_dep::DOUBLE / all_dep ELSE 0 END AS morning_share,
               morn_dep::DOUBLE / days AS mean_morning_dep,
               CASE WHEN (morn_arr + morn_dep) > 0
                    THEN (morn_arr::DOUBLE - morn_dep) / (morn_arr + morn_dep) ELSE 0 END AS net_balance
          FROM agg
         WHERE days > 0
         ORDER BY name
        """
    ).df()

    # A station with a genuinely tiny level would give log() a value near -inf and
    # a Poisson offset that swamps every tree. The floor is well below the
    # MIN_MEAN_MORNING_DEP filter, so in practice it never binds -- it is here so
    # that a future loosening of that filter cannot produce silent infinities.
    df["log_level"] = df["mean_morning_dep"].clip(lower=0.05).map(math.log)
    return df[["name", "days", "morning_share", "log_level", "net_balance", "mean_morning_dep"]]


def store(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, table: str = "station_profile") -> None:
    con.register("_sp", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _sp")
    con.unregister("_sp")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upto", default=None, help="only use station-days strictly before this YYYY-MM-DD")
    a = ap.parse_args(argv)
    with connect() as con:
        df = compute(con, upto=a.upto)
        store(con, df)
        print(f"[stations] {len(df)} profiles")
        print(df.describe().loc[["mean", "min", "max"]].round(3).to_string())
        row = df[df.name == paths.STATION_345]
        if row.empty:
            raise SystemExit(f"station {paths.STATION_345!r} missing from profiles")
        r = row.iloc[0]
        print(
            f"[stations] 345: days={int(r.days)} morning_share={r.morning_share:.3f} "
            f"log_level={r.log_level:.3f} (mean {r.mean_morning_dep:.2f}/morning) net_balance={r.net_balance:+.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
