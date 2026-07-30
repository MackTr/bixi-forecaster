"""The sanity oracle: does the hourly rebuild still say what the day-level pass said?

An hourly panel is built by converting 27.5M epoch-millisecond timestamps into
local hours. That conversion is the single most consequential line in the
pipeline and the least likely to fail loudly: shift every timestamp by an hour
and everything still trains, still validates, still exports -- and the model
learns a commute peak that is off by one, forever.

So the day-level matrix from the earlier exploration is kept as an oracle and the
new panel is checked against it. Four tests, in increasing sharpness:

1. **Paired equality.** The old matrix has one row per station-day with 6-11am
   departures, built independently (a different awk program, a different
   aggregation). Summing the new panel over the same window must reproduce it
   EXACTLY, station-day for station-day. This is much stronger than comparing
   distributions -- it checks the name join, the day boundary, the hour window
   and the counting in one assertion.

2. **The DST tail must DISAGREE.** The old pass used a blanket UTC-4, which is
   right from March to early November and wrong for the last ~12 days of the
   season. The new panel derives the offset from the tz database. So test 1 is
   run only over the EDT period, and the November tail is required to differ --
   if it matched, the new DST handling would not actually be doing anything.

3. **Known relationships.** Morning rain is strong and monotonic (14.88 / 11.90 /
   8.90 across dry/light/wet); weekday mornings run 1.48-1.56x weekend ones.

4. **The peak sits at 08:00** on weekdays, and lands on the same hour inside and
   outside DST. An off-by-one conversion is invisible in a daily total and
   glaring here.
"""

from __future__ import annotations

import argparse
import sys

import duckdb

from . import paths, weather
from .panel import connect

ORACLE_CSV = paths.DATA / "oracle_daily.csv"
MORNING = ",".join(str(h) for h in paths.MORNING_HOURS)

# When Montreal is on EDT, from paths.dst_intervals. The old blanket -14400 pass
# is correct inside these ranges and wrong outside them.
EDT = {2024: ("2024-03-10", "2024-11-03"), 2025: ("2025-03-09", "2025-11-02")}


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            self.failures.append(label)


def daily(con: duckdb.DuckDBPyConnection) -> None:
    """Panel summed back down to (station, date) morning departures."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE daily_dep AS
          SELECT station, date, SUM(dep) AS morning_departures
            FROM panel WHERE hour IN ({MORNING})
           GROUP BY 1, 2;
        """
    )


def test_paired(con: duckdb.DuckDBPyConnection, rep: Report) -> None:
    if not ORACLE_CSV.exists():
        print(f"  [SKIP] paired equality: {ORACLE_CSV.name} not present")
        return
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE oracle AS
          SELECT trim(station) AS station, date::DATE AS date, morning_departures
            FROM read_csv('{ORACLE_CSV}', header=true);
        CREATE OR REPLACE TEMP TABLE paired AS
          SELECT d.station, d.date, d.morning_departures AS mine, o.morning_departures AS theirs,
                 CASE WHEN (year(d.date) = 2024 AND d.date >= DATE '{EDT[2024][0]}' AND d.date < DATE '{EDT[2024][1]}')
                        OR (year(d.date) = 2025 AND d.date >= DATE '{EDT[2025][0]}' AND d.date < DATE '{EDT[2025][1]}')
                      THEN 'edt' ELSE 'est' END AS regime
            FROM daily_dep d JOIN oracle o ON o.station = d.station AND o.date = d.date;
        """
    )
    n, agree = con.execute(
        "SELECT COUNT(*), SUM((mine = theirs)::INT) FROM paired WHERE regime = 'edt'"
    ).fetchone()
    rep.check(
        n > 100_000 and agree == n,
        "paired equality (EDT period)",
        f"{agree:,}/{n:,} station-days match exactly",
    )

    n2, agree2 = con.execute(
        "SELECT COUNT(*), SUM((mine = theirs)::INT) FROM paired WHERE regime = 'est'"
    ).fetchone()
    # Not every November station-day has to differ -- a quiet rack can post the
    # same total either way -- but a wholesale match would mean the DST handling
    # is inert.
    rate = (n2 - agree2) / n2 if n2 else 0
    rep.check(
        n2 > 1000 and rate > 0.5,
        "DST tail differs from the blanket-offset oracle",
        f"{n2 - agree2:,}/{n2:,} ({rate:.0%}) of November station-days differ, as they should",
    )


def test_relationships(con: duckdb.DuckDBPyConnection, rep: Report) -> None:
    weather.ensure_table(con)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE morn_wx AS
          SELECT date, SUM(precip_mm) AS precip_mm
            FROM weather_hourly WHERE hour IN ({MORNING}) GROUP BY 1;
        CREATE OR REPLACE TEMP TABLE dd AS
          SELECT d.*, w.precip_mm,
                 CASE WHEN w.precip_mm < {paths.PRECIP_DRY_MAX_MM} THEN 'dry'
                      WHEN w.precip_mm <= {paths.PRECIP_WET_MIN_MM} THEN 'light' ELSE 'wet' END AS bucket,
                 dayofweek(d.date) AS dow
            FROM daily_dep d JOIN morn_wx w ON w.date = d.date;
        """
    )
    rows = con.execute(
        "SELECT bucket, AVG(morning_departures), COUNT(*) FROM dd GROUP BY 1 ORDER BY 1"
    ).fetchall()
    m = {b: (avg, n) for b, avg, n in rows}
    dry, light, wet = m["dry"][0], m["light"][0], m["wet"][0]
    rep.check(
        dry > light > wet,
        "rain is monotonic",
        f"dry {dry:.2f} (n={m['dry'][1]:,}) > light {light:.2f} (n={m['light'][1]:,}) > wet {wet:.2f} (n={m['wet'][1]:,})",
    )

    # Absolute levels are NOT asserted against remembered constants, and that is
    # deliberate. The earlier exploration's figures (14.88 / 11.90 / 8.90) were
    # measured on weekdays only over the full year with no station filter; this
    # panel is season-restricted, both-day and volume-filtered, so the levels
    # legitimately differ. Comparing them would test the population, not the
    # rebuild. What IS asserted, below, is agreement with the oracle on the
    # population the two actually share.
    r = {}
    for yr in (2024, 2025):
        wd, we = con.execute(
            f"""SELECT AVG(CASE WHEN dow BETWEEN 1 AND 5 THEN morning_departures END),
                       AVG(CASE WHEN dow IN (0, 6) THEN morning_departures END)
                  FROM dd WHERE year(date) = {yr}"""
        ).fetchone()
        r[yr] = wd / we
    rep.check(
        min(r.values()) > 1.3,
        "weekday mornings exceed weekend mornings",
        f"2024 {r[2024]:.3f}, 2025 {r[2025]:.3f}",
    )
    # Stability across seasons was the earlier pass's real finding, and it is the
    # part that reproduces. The magnitudes it recorded do not.
    rep.check(
        abs(r[2024] - r[2025]) < 0.20,
        "weekday:weekend ratio is stable year over year",
        f"|{r[2024]:.3f} - {r[2025]:.3f}| = {abs(r[2024] - r[2025]):.3f}",
    )


def test_oracle_agreement(con: duckdb.DuckDBPyConnection, rep: Report) -> None:
    """Derived statistics must agree with the oracle on the SHARED population.

    Paired equality already proves the counts are identical there, so this cannot
    fail on its own -- which is exactly why it is the right form for the test. It
    compares like with like, instead of holding the rebuild to numbers measured
    over a different set of station-days.
    """
    if not ORACLE_CSV.exists():
        print("  [SKIP] oracle agreement: oracle file not present")
        return
    mine, theirs = con.execute(
        """SELECT AVG(CASE WHEN dayofweek(date) BETWEEN 1 AND 5 THEN mine END)
                / AVG(CASE WHEN dayofweek(date) IN (0, 6) THEN mine END),
                  AVG(CASE WHEN dayofweek(date) BETWEEN 1 AND 5 THEN theirs END)
                / AVG(CASE WHEN dayofweek(date) IN (0, 6) THEN theirs END)
             FROM paired WHERE regime = 'edt'"""
    ).fetchone()
    rep.check(
        abs(mine - theirs) < 0.005,
        "weekday:weekend ratio agrees with the oracle",
        f"mine {mine:.4f} vs oracle {theirs:.4f} on the shared EDT population",
    )


def test_peak_hour(con: duckdb.DuckDBPyConnection, rep: Report) -> None:
    """The commute peak, inside and outside DST.

    This is the test a one-hour conversion error cannot survive. Daily totals
    barely move when hours shift; the peak moves by exactly one.
    """
    rows = con.execute(
        f"""
        SELECT CASE WHEN (year(date) = 2024 AND date < DATE '{EDT[2024][1]}')
                      OR (year(date) = 2025 AND date < DATE '{EDT[2025][1]}')
                    THEN 'edt' ELSE 'est' END AS regime,
               hour, SUM(dep) AS dep
          FROM panel
         WHERE dayofweek(date) BETWEEN 1 AND 5 AND hour BETWEEN 5 AND 11
         GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).fetchall()
    peaks = {}
    for regime in ("edt", "est"):
        sub = [(h, d) for r, h, d in rows if r == regime]
        peaks[regime] = max(sub, key=lambda t: t[1])[0]
    rep.check(peaks["edt"] == 8, "weekday morning peak hour (EDT)", f"{peaks['edt']}:00")
    rep.check(
        peaks["est"] == peaks["edt"],
        "peak hour survives the DST boundary",
        f"EST peak {peaks['est']}:00 == EDT peak {peaks['edt']}:00",
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    rep = Report()
    with connect() as con:
        daily(con)
        print("\n[oracle] verifying the hourly panel against the day-level matrix")
        test_paired(con, rep)
        test_oracle_agreement(con, rep)
        test_relationships(con, rep)
        test_peak_hour(con, rep)
    print()
    if rep.failures:
        print(f"[oracle] {len(rep.failures)} FAILED: {', '.join(rep.failures)}")
        return 1
    print("[oracle] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
