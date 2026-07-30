"""Step 3: the dense (station, date, hour) panel.

The exploratory pass built a station-DAY matrix, 402,092 rows. That matrix cannot
produce a run-out TIME -- a 6-11am total says how many bikes leave, not when the
rack empties -- so this rebuilds at hourly granularity. The day-level numbers it
implies are not discarded, though: backtest.py sums this panel back down to days
and checks it reproduces them. A broken UTC-ms -> local-hour conversion is
invisible in a trained model and obvious in that comparison.

Two decisions carry most of the weight here:

**Zeros are data, and false zeros are poison.** A Poisson model has to see the
hours where nothing happened, so the grid is densified rather than left sparse.
But a station that had not been installed yet, or was pulled for construction,
also produces no rows -- and filling those with zeros would teach the model that
real stations sit idle. So a station-day is kept only if the station shows
activity within a +/-3 day window and the date falls inside that station's
observed service range for the year. Quiet days survive; nonexistent ones do not.

**Only the hours the simulation reads.** The Monte Carlo starts at 22:00 and runs
to noon, so 13:00-21:00 is never consumed. Dropping it removes ~38% of the rows
at zero cost to anything downstream.
"""

from __future__ import annotations

import argparse
import sys
import time

import duckdb

from . import paths

DB = paths.DATA / "bixi.duckdb"

# A station-day needs activity within this many days either side to count as in
# service. Wide enough to bridge a quiet shoulder-season week, narrow enough that
# an install or removal is not smeared across a month.
SERVICE_WINDOW_DAYS = 3
# Stations quieter than this over the morning window have a log level too noisy
# to serve as a Poisson offset -- the offset would be fitting sampling noise.
MIN_MEAN_MORNING_DEP = 1.0
# ...and so do stations that barely existed. A rack installed in late October has
# a level estimated from a handful of cold days; it would enter training with a
# confidently wrong offset. 95 of 1,268 stations fall below this line.
MIN_SERVICE_DAYS = 30


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB))


def build(con: duckdb.DuckDBPyConnection, years: tuple[int, ...]) -> None:
    t0 = time.time()

    # --- raw projections -> per-(station, day, hour) event counts -------------
    # Departures and arrivals are aggregated separately and then outer-joined.
    # A UNION ALL of 55M tagged rows would give the same answer through a much
    # larger intermediate.
    parts = []
    for y in years:
        trips = paths.DATA / f"trips_{y}.tsv"
        dic = paths.DATA / f"stations_{y}.tsv"
        if not trips.exists():
            raise SystemExit(f"missing {trips} -- run `python -m bixi_train.opendata` first")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE t_{y} AS
              SELECT * FROM read_csv('{trips}', delim='\t', header=false, columns={{
                's_id':'INTEGER','s_day':'INTEGER','s_hour':'TINYINT',
                'e_id':'INTEGER','e_day':'INTEGER','e_hour':'TINYINT'}});
            CREATE OR REPLACE TEMP TABLE d_{y} AS
              SELECT * FROM read_csv('{dic}', delim='\t', header=false,
                columns={{'id':'INTEGER','name':'VARCHAR'}}, quote='');
            """
        )
        # Station names are the only join key open data gives us, and they carry
        # accents and stray whitespace that can differ between years, so they are
        # trimmed on the way in.
        parts.append(
            f"""
            SELECT COALESCE(dp.name, ar.name) AS name,
                   COALESCE(dp.day,  ar.day)  AS day,
                   COALESCE(dp.hour, ar.hour) AS hour,
                   COALESCE(dp.n, 0) AS dep, COALESCE(ar.n, 0) AS arr
            FROM (SELECT trim(d.name) AS name, t.s_day AS day, t.s_hour AS hour, COUNT(*) AS n
                    FROM t_{y} t JOIN d_{y} d ON d.id = t.s_id
                   GROUP BY 1, 2, 3) dp
            FULL OUTER JOIN
                 (SELECT trim(d.name) AS name, t.e_day AS day, t.e_hour AS hour, COUNT(*) AS n
                    FROM t_{y} t JOIN d_{y} d ON d.id = t.e_id
                   WHERE t.e_day >= 0
                   GROUP BY 1, 2, 3) ar
            ON ar.name = dp.name AND ar.day = dp.day AND ar.hour = dp.hour
            """
        )

    # Regrouped after the union: a trip in the 2025 file can start on a 2024 date
    # (New Year's Eve crossing midnight), so the per-year parts are not quite
    # disjoint on (name, day, hour).
    con.execute(
        "CREATE OR REPLACE TABLE events AS SELECT name, day, hour, SUM(dep) AS dep, SUM(arr) AS arr FROM ("
        + " UNION ALL ".join(parts)
        + ") GROUP BY 1, 2, 3"
    )
    n_events = con.sql("SELECT COUNT(*) FROM events").fetchone()[0]
    print(f"[panel] {n_events:,} station-day-hour event cells ({time.time() - t0:.0f}s)", flush=True)

    # --- which station-days actually existed ---------------------------------
    con.execute(
        f"""
        CREATE OR REPLACE TABLE day_activity AS
          SELECT name, day, DATE '1970-01-01' + day AS date, SUM(dep + arr) AS n
            FROM events GROUP BY 1, 2, 3;

        CREATE OR REPLACE TABLE station_days AS
        WITH span AS (
          -- Per station-YEAR, so a station added mid-2025 is not treated as
          -- having existed since April 2024.
          SELECT name, year(date) AS yr, MIN(day) AS first_day, MAX(day) AS last_day
            FROM day_activity GROUP BY 1, 2
        ), cal AS (
          -- Every calendar day inside each station's own service span, including
          -- the days with no activity at all. Those are the rows the window
          -- below has to judge; building the calendar from day_activity instead
          -- would make the test vacuous, since every row there is active by
          -- construction.
          -- range() yields BIGINT and DuckDB only defines +(DATE, INTEGER),
          -- so the cast is load-bearing rather than cosmetic.
          SELECT s.name, d.day::INTEGER AS day, DATE '1970-01-01' + d.day::INTEGER AS date
            FROM span s, UNNEST(range(s.first_day, s.last_day + 1)) AS d(day)
        ), scored AS (
          SELECT c.name, c.day, c.date,
                 SUM(COALESCE(a.n, 0)) OVER (PARTITION BY c.name ORDER BY c.day
                                RANGE BETWEEN {SERVICE_WINDOW_DAYS} PRECEDING
                                          AND {SERVICE_WINDOW_DAYS} FOLLOWING) AS nearby
            FROM cal c
            LEFT JOIN day_activity a ON a.name = c.name AND a.day = c.day
        )
        SELECT name, day, date
          FROM scored
         WHERE nearby > 0
           AND (month(date) * 100 + day(date)) BETWEEN {paths.SEASON_START_MD[0] * 100 + paths.SEASON_START_MD[1]}
                                                   AND {paths.SEASON_END_MD[0] * 100 + paths.SEASON_END_MD[1]}
           AND year(date) IN ({','.join(str(y) for y in years)});
        """
    )

    # --- volume filter -------------------------------------------------------
    hours = ",".join(str(h) for h in paths.MORNING_HOURS)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE station_volume AS
          SELECT sd.name,
                 COUNT(DISTINCT sd.day) AS days,
                 SUM(COALESCE(e.dep, 0)) / COUNT(DISTINCT sd.day) AS mean_morning_dep
            FROM station_days sd
            LEFT JOIN events e ON e.name = sd.name AND e.day = sd.day AND e.hour IN ({hours})
           GROUP BY 1;
        CREATE OR REPLACE TABLE active_stations AS
          SELECT name FROM station_volume
           WHERE mean_morning_dep >= {MIN_MEAN_MORNING_DEP} AND days >= {MIN_SERVICE_DAYS};
        """
    )
    tot, act = con.sql(
        "SELECT (SELECT COUNT(*) FROM station_volume), (SELECT COUNT(*) FROM active_stations)"
    ).fetchone()
    print(
        f"[panel] {act:,}/{tot:,} stations clear {MIN_MEAN_MORNING_DEP} mean morning departures "
        f"and {MIN_SERVICE_DAYS} service days",
        flush=True,
    )

    # --- the dense grid ------------------------------------------------------
    panel_hours = ",".join(f"({h})" for h in paths.PANEL_HOURS)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE panel AS
          SELECT sd.name AS station, sd.date, h.hour,
                 COALESCE(e.dep, 0)::INTEGER AS dep,
                 COALESCE(e.arr, 0)::INTEGER AS arr
            FROM station_days sd
            JOIN active_stations a ON a.name = sd.name
           CROSS JOIN (VALUES {panel_hours}) AS h(hour)
            LEFT JOIN events e
              ON e.name = sd.name AND e.day = sd.day AND e.hour = h.hour;
        """
    )
    rows, stations, d0, d1 = con.sql(
        "SELECT COUNT(*), COUNT(DISTINCT station), MIN(date), MAX(date) FROM panel"
    ).fetchone()
    zero = con.sql("SELECT AVG((dep = 0)::INT) FROM panel").fetchone()[0]
    print(
        f"[panel] {rows:,} rows | {stations:,} stations | {d0}..{d1} | {zero:.1%} of hours have zero departures "
        f"({time.time() - t0:.0f}s total)",
        flush=True,
    )

    # Station 345 must survive every filter above -- it is the only station this
    # service actually predicts, and a silent drop would surface as a confusing
    # failure much later in export.
    n345 = con.execute("SELECT COUNT(*) FROM panel WHERE station = ?", [paths.STATION_345]).fetchone()[0]
    if n345 == 0:
        raise SystemExit(f"station {paths.STATION_345!r} is absent from the panel -- check the name join")
    print(f"[panel] station 345 ({paths.STATION_345}): {n345:,} rows")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=list(paths.TRAIN_YEARS))
    a = ap.parse_args(argv)
    paths.ensure_dirs()
    with connect() as con:
        build(con, tuple(a.years))
    return 0


if __name__ == "__main__":
    sys.exit(main())
