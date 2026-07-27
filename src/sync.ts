// Digests bixi-monitor's observation stream into hourly facts.
//
// bixi-predictor collapses a whole day into one row, which is enough to ask
// "did it run out, and when". It is not enough to TRAIN or SIMULATE: a 6-11am
// total cannot produce a time. So the same observation walk is re-cut at hourly
// granularity, and — the substantive addition — every inventory change is
// classified as truck or organic.
//
// That split is load-bearing. BIXI's open trip data has no inventory column, so
// trucks are structurally invisible in it; a demand model trained on trips can
// only ever learn organic flow. Keeping the two apart here is what lets the
// simulation add trucks back as an explicit, separately-estimated term instead
// of silently blaming the demand model for a rebalancing van.
//
// The run-out definition is deliberately IDENTICAL to bixi-predictor's (first
// usable-bikes >0 -> <=0 transition of the local day, from a stream that starts
// 2h early so the previous value is known at midnight). If it ever drifts, the
// A/B stops comparing models and starts comparing definitions.

import type { Env } from "./worker";
import { TZ, addDays, bucketParts, dayDiff, dowOf, localParts, localToday, midnightEpoch, wallToEpoch } from "./tz";
import { holidayName } from "./holidays";
import { fetchObservations } from "./monitor";

// Truck classification, and it is NOT the rule bixi-predictor uses.
//
// The sibling calls any same-direction run totalling >=5 bikes a truck. That
// threshold was measured against 14 days of this station's observations before
// being adopted here, and it does not survive:
//
//   burst size |1|..|4| decays geometrically with ratio 0.393 — the unambiguous
//   organic-clustering regime. Extrapolating that fit predicts ~32 events in
//   the 5-8 band; 41 were observed. So the 5-8 band is essentially ALL organic
//   clustering. The genuine truck population is a separate cluster at >=9,
//   where the same fit predicts under one event and twelve occurred.
//
// Rate confirms it independently, which is why it is trusted: bursts >=11 run
// at a median 2.20 bikes/min — the pace a van physically loads at — while the
// 5-8 band runs at 0.78/min, SLOWER than the isolated 1-4 changes (1.00/min).
// Slower than a commuter is the opposite of a truck.
//
// This matters in a specific, directional way rather than academically:
// organic_out is what the nightly level calibration fits against, so labelling
// real commuters as trucks removes them from the fit, drives the calibrated
// level DOWN, and makes every run-out prediction systematically LATE. It also
// inflated p_visit by 2.7x (56 detections vs 21), which would have injected
// invented truck variance into every simulated window.
const BURST_GAP = 300; // seconds
// Size alone is decisive here — 100x excess over the organic tail.
const TRUCK_DECISIVE_BIKES = 9;
// Below that, a burst needs truck PACE as well as truck size.
const TRUCK_MIN_BIKES = 5;
const TRUCK_MIN_RATE = 1.5; // bikes per minute

// A single-step change carries no rate information at all (the monitor may
// simply have missed the intermediate states), so it is judged on size alone
// rather than being handed an infinite rate.
function isTruck(sum: number, spanSeconds: number, steps: number): boolean {
  const size = Math.abs(sum);
  if (size >= TRUCK_DECISIVE_BIKES) return true;
  if (size < TRUCK_MIN_BIKES || steps < 2 || spanSeconds <= 0) return false;
  return (size / spanSeconds) * 60 >= TRUCK_MIN_RATE;
}

export interface SyncedDay {
  date: string;
  runoutMinutes: number | null;
  obsCount: number;
  complete: boolean;
  eveningBikes: number | null;
  eveningSwept: number | null;
  hours: number; // hourly_facts rows written for this date
}

interface HourCell {
  date: string;
  hour: number;
  open: number | null;
  min: number | null;
  close: number | null;
  emptySeconds: number;
  truckIn: number;
  truckOut: number;
  organicIn: number;
  organicOut: number;
  obs: number;
}

// America/Toronto is always a whole number of hours from UTC (-5 / -4), so a UTC
// hour boundary IS a local hour boundary and this needs no Intl call. The same
// assumption underpins the offline panel build.
//
// DST edge, accepted rather than engineered around: on the fall-back night local
// hour 1 happens twice and the two passes merge into one (date, hour) row; on
// the spring-forward night local hour 2 has no row at all. Both are 1-3am, far
// from the 6-11am window this model is about, and pretending otherwise would
// cost a composite key nothing else wants.
const hourEndOf = (t: number) => Math.floor(t / 3600) * 3600 + 3600;

export async function syncDays(env: Env, dates: string[], now: number): Promise<SyncedDay[]> {
  const today = localToday(now);
  const out: SyncedDay[] = [];

  for (const run of contiguousRuns(dates, 5)) {
    const runStart = midnightEpoch(run[0]);
    const endEpoch = midnightEpoch(addDays(run[run.length - 1], 1));
    // Start 2h early so the held value is known at the first midnight (the
    // monitor heartbeats at least every 15 min); stop 1s short of the next
    // midnight so an exactly-on-boundary row can't be counted into the last day.
    const from = runStart - 7200;
    const to = Math.min(now, endEpoch - 1);
    if (to <= from) continue;
    const rows = await fetchObservations(env, from, to);
    if (!rows.length) continue;

    const inRun = new Set(run);
    const cells = new Map<string, HourCell>();
    const cellFor = (date: string, hour: number, heldValue: number | null): HourCell | null => {
      if (!inRun.has(date)) return null; // lead-in, or spill past the run
      const key = `${date}|${hour}`;
      let c = cells.get(key);
      if (!c) {
        c = {
          date,
          hour,
          open: heldValue,
          min: heldValue,
          close: heldValue,
          emptySeconds: 0,
          truckIn: 0,
          truckOut: 0,
          organicIn: 0,
          organicOut: 0,
          obs: 0,
        };
        cells.set(key, c);
      }
      return c;
    };

    // Between two observations the station holds its last value. Walking that
    // interval hour by hour is what fills hours with no observations at all
    // (open/min/close held, obs_count 0) and what makes empty_minutes exact
    // across an hour boundary — a rack that empties at 07:50 and is refilled at
    // 08:20 owes 10 minutes to hour 7 and 20 to hour 8.
    const hold = (t0: number, t1: number, v: number) => {
      let t = Math.max(t0, runStart);
      while (t < t1) {
        // bucketParts, not localParts: this runs per observation per hour
        // segment, and constructing the wall-clock facts uncached is precisely
        // what blew the CPU budget in bixi-monitor (see tz.ts). Only dateStr
        // and hour are read, both constant within a 15-minute bucket.
        const p = bucketParts(t, TZ);
        const segEnd = Math.min(t1, hourEndOf(t));
        const c = cellFor(p.dateStr, p.hour, v);
        if (c) {
          if (v <= 0) c.emptySeconds += segEnd - t;
          c.min = c.min == null ? v : Math.min(c.min, v);
          c.close = v;
        }
        t = segEnd;
      }
    };

    // A burst is only known to BE a burst once it closes, so deltas are held
    // with the hour cell they belong to and attributed on close. Without the
    // per-item hours, a truck visit spanning 07:58-08:02 would dump its whole
    // volume into one hour.
    interface Burst {
      dir: 1 | -1;
      sum: number;
      startTs: number;
      lastTs: number;
      items: { cell: HourCell | null; delta: number }[];
    }
    let burst: Burst | null = null;
    const closeBurst = () => {
      if (!burst) return;
      const truck = isTruck(burst.sum, burst.lastTs - burst.startTs, burst.items.length);
      for (const { cell, delta } of burst.items) {
        if (!cell) continue;
        if (delta > 0) truck ? (cell.truckIn += delta) : (cell.organicIn += delta);
        else truck ? (cell.truckOut += -delta) : (cell.organicOut += -delta);
      }
      burst = null;
    };

    const eveningEpochs = run.map((d) => {
      const [y, m, dd] = d.split("-").map(Number);
      return wallToEpoch(y, m, dd, 22, 0, TZ);
    });
    const daily = new Map(
      run.map((d) => [d, { runoutEpoch: null as number | null, eveningBikes: null as number | null }]),
    );

    let prev: number | null = null;
    let prevTs = 0;
    for (const r of rows) {
      if (prev != null) hold(prevTs, r.ts, prev);

      const p = bucketParts(r.ts, TZ);
      const cell = cellFor(p.dateStr, p.hour, prev);
      if (cell) {
        cell.obs++;
        cell.close = r.bikes;
        cell.min = cell.min == null ? r.bikes : Math.min(cell.min, r.bikes);
      }

      const day = daily.get(p.dateStr);
      if (day) {
        // Identical to bixi-predictor: the day's FIRST >0 -> <=0 transition.
        if (prev != null && prev > 0 && r.bikes <= 0 && day.runoutEpoch == null) day.runoutEpoch = r.ts;
        const i = run.indexOf(p.dateStr);
        if (i >= 0 && r.ts <= eveningEpochs[i]) day.eveningBikes = r.bikes;
      }

      if (prev != null && r.bikes !== prev) {
        const delta = r.bikes - prev;
        const dir = delta > 0 ? (1 as const) : (-1 as const);
        if (burst && burst.dir === dir && r.ts - burst.lastTs <= BURST_GAP) {
          burst.sum += delta;
          burst.lastTs = r.ts;
          burst.items.push({ cell, delta });
        } else {
          closeBurst();
          burst = { dir, sum: delta, startTs: r.ts, lastTs: r.ts, items: [{ cell, delta }] };
        }
      }
      prev = r.bikes;
      prevTs = r.ts;
    }
    closeBurst();
    // Hold the final value out to the end of the covered range, so the current
    // hour isn't silently truncated at the last observation.
    if (prev != null) hold(prevTs, to, prev);

    const statements: D1PreparedStatement[] = [];
    const hourStmt = env.DB.prepare(
      `INSERT INTO hourly_facts (date, hour, bikes_open, bikes_min, bikes_close, empty_minutes,
                                 truck_in, truck_out, organic_in, organic_out, obs_count, synced_ts)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(date, hour) DO UPDATE SET
         bikes_open = excluded.bikes_open, bikes_min = excluded.bikes_min,
         bikes_close = excluded.bikes_close, empty_minutes = excluded.empty_minutes,
         truck_in = excluded.truck_in, truck_out = excluded.truck_out,
         organic_in = excluded.organic_in, organic_out = excluded.organic_out,
         obs_count = excluded.obs_count, synced_ts = excluded.synced_ts`,
    );
    const perDate = new Map(run.map((d) => [d, { hours: 0, obs: 0, swept: 0 }]));
    for (const c of cells.values()) {
      const agg = perDate.get(c.date)!;
      agg.hours++;
      agg.obs += c.obs;
      // Evening sweep = bikes trucked out 17:00-21:59, the window bixi-predictor
      // uses. Derived from the hourly split rather than re-detected, so the two
      // numbers cannot disagree within this repo.
      //
      // This WILL read lower than bixi-predictor's evening_swept for the same
      // date, because the classifier above is stricter than the sibling's. That
      // divergence is deliberate and harmless: nothing in this service consumes
      // evening_swept (the simulation seeds from evening_bikes, an observation
      // rather than an inference), and the column that actually has to agree
      // between the two services — runout_minutes — is computed identically and
      // verified to match.
      if (c.hour >= 17 && c.hour < 22) agg.swept += c.truckOut;
      statements.push(
        hourStmt.bind(
          c.date,
          c.hour,
          c.open,
          c.min,
          c.close,
          Math.round(c.emptySeconds / 60),
          c.truckIn,
          c.truckOut,
          c.organicIn,
          c.organicOut,
          c.obs,
          now,
        ),
      );
    }

    const dayStmt = env.DB.prepare(
      `INSERT INTO daily_facts (date, dow, is_holiday, runout_minutes, evening_bikes, evening_swept,
                                obs_count, complete, synced_ts)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(date) DO UPDATE SET
         dow = excluded.dow, is_holiday = excluded.is_holiday,
         runout_minutes = excluded.runout_minutes,
         evening_bikes = COALESCE(excluded.evening_bikes, daily_facts.evening_bikes),
         evening_swept = COALESCE(excluded.evening_swept, daily_facts.evening_swept),
         obs_count = excluded.obs_count, complete = excluded.complete,
         synced_ts = excluded.synced_ts`,
    );
    for (const date of run) {
      const agg = perDate.get(date)!;
      const st = daily.get(date)!;
      const runoutMinutes =
        st.runoutEpoch != null
          ? (() => {
              const p = localParts(st.runoutEpoch, TZ);
              return p.hour * 60 + p.minute;
            })()
          : null;
      const complete = date < today ? 1 : 0;
      const swept = agg.obs > 0 ? agg.swept : null; // meaningless if the walk saw nothing
      statements.push(
        dayStmt.bind(date, dowOf(date), holidayName(date) ? 1 : 0, runoutMinutes, st.eveningBikes, swept, agg.obs, complete, now),
      );
      out.push({
        date,
        runoutMinutes,
        obsCount: agg.obs,
        complete: !!complete,
        eveningBikes: st.eveningBikes,
        eveningSwept: swept,
        hours: agg.hours,
      });
    }

    // Hourly granularity means ~24x bixi-predictor's write volume, so this must
    // never become a per-row `await .run()` loop — one batch is one round trip.
    await writeBatched(env, statements);
  }
  return out;
}

// D1 caps how much one batch may carry; 50 statements per round trip keeps a
// full 14-day repair (~350 rows) to a handful of calls without ever approaching
// the limit.
export async function writeBatched(env: Env, statements: D1PreparedStatement[], size = 50): Promise<void> {
  for (let i = 0; i < statements.length; i += size) {
    await env.DB.batch(statements.slice(i, i + size));
  }
}

// Dates in [today - lookbackDays, today] that are missing or were synced before
// their day ended. Yesterday and today always qualify (today is always fresh;
// yesterday was complete=0 when synced at 10pm yesterday), so a normal night
// touches 2 days ≈ 48 hourly rows, not the full window.
export async function datesNeedingSync(env: Env, lookbackDays: number, today: string): Promise<string[]> {
  const from = addDays(today, -lookbackDays);
  const res = await env.DB.prepare(`SELECT date FROM daily_facts WHERE date >= ? AND date <= ? AND complete = 1`)
    .bind(from, today)
    .all<{ date: string }>();
  const done = new Set((res.results ?? []).map((r) => r.date));
  const out: string[] = [];
  for (let d = from; d <= today; d = addDays(d, 1)) {
    if (!done.has(d) || dayDiff(d, today) <= 1) out.push(d);
  }
  return out;
}

// Split sorted dates into contiguous runs of at most maxLen, so each run is one
// bounded monitor fetch (a busy on-change day is a few hundred rows; 5 days
// stays far under the API's 20k cap).
function contiguousRuns(dates: string[], maxLen: number): string[][] {
  const sorted = [...new Set(dates)].sort();
  const runs: string[][] = [];
  for (const d of sorted) {
    const cur = runs[runs.length - 1];
    if (cur && cur.length < maxLen && dayDiff(cur[cur.length - 1], d) === 1) cur.push(d);
    else runs.push([d]);
  }
  return runs;
}
