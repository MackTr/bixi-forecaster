// The truck prior: P(a rebalancing visit | dow, hour) and how many bikes it
// moves when it happens.
//
// This table exists because BIXI's open trip data cannot supply it. Trips have
// no inventory column, so a van that removes 12 bikes leaves no trace in the
// 27.5M-row training set — yet at this station trucks move on the order of
// 13.5 bikes a day in summer, which is most of a 19-dock rack. A demand model
// trained on trips alone would be asked to explain variance it has no access to.
//
// Two deliberate limits, both stated rather than hidden:
//
//  1. Seeding the simulation with the ACTUAL 10pm count absorbs the evening
//     sweep entirely — by far the largest truck event of the day is already
//     accounted for by the initial condition, and only overnight and morning
//     moves need modelling here.
//  2. Six weeks of observations is thin for a 168-cell table. So this stores the
//     raw empirical counts plus n_days and lets the CONSUMER shrink toward the
//     pooled rate (see simulate.ts). Rebalancing is an irreducible variance
//     floor: the honest response to a thin cell is a wider window, not a
//     confident point estimate.

import type { Env } from "./worker";
import { addDays } from "./tz";
import { writeBatched } from "./sync";

// How far back to build the profile. Long enough to accumulate cells, short
// enough that a seasonal change in the truck schedule isn't averaged with the
// present one.
const PROFILE_DAYS = 56;

interface Cell {
  dow: number;
  hour: number;
  visits: number;
  days: number;
  sum: number;
  sumSq: number;
}

export async function buildRebalanceProfile(env: Env, today: string, now: number): Promise<number> {
  const from = addDays(today, -PROFILE_DAYS);
  // Only hours from days the walk actually covered: an unobserved hour is not
  // evidence of "no truck came", and counting it as such would bias every
  // p_visit toward zero.
  const res = await env.DB.prepare(
    `SELECT h.date, h.hour, h.truck_in, h.truck_out, f.dow
     FROM hourly_facts h JOIN daily_facts f ON f.date = h.date
     WHERE h.date >= ? AND h.date < ? AND h.obs_count > 0`,
  )
    .bind(from, today)
    .all<{ date: string; hour: number; truck_in: number; truck_out: number; dow: number }>();
  const rows = res.results ?? [];
  if (!rows.length) return 0;

  const cells = new Map<string, Cell>();
  for (const r of rows) {
    const key = `${r.dow}|${r.hour}`;
    let c = cells.get(key);
    if (!c) cells.set(key, (c = { dow: r.dow, hour: r.hour, visits: 0, days: 0, sum: 0, sumSq: 0 }));
    c.days++;
    const delta = r.truck_in - r.truck_out;
    if (r.truck_in > 0 || r.truck_out > 0) {
      c.visits++;
      c.sum += delta;
      c.sumSq += delta * delta;
    }
  }

  const stmt = env.DB.prepare(
    `INSERT INTO rebalance_profile (dow, hour, p_visit, mean_delta, sd_delta, n_days, built_ts)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(dow, hour) DO UPDATE SET
       p_visit = excluded.p_visit, mean_delta = excluded.mean_delta,
       sd_delta = excluded.sd_delta, n_days = excluded.n_days, built_ts = excluded.built_ts`,
  );
  const statements = [...cells.values()].map((c) => {
    const mean = c.visits ? c.sum / c.visits : 0;
    // Population sd over visits; a single observed visit has no spread of its
    // own, so it reports 0 here and inherits the pooled sd downstream.
    const varr = c.visits > 1 ? Math.max(0, c.sumSq / c.visits - mean * mean) : 0;
    return stmt.bind(c.dow, c.hour, c.visits / c.days, mean, Math.sqrt(varr), c.days, now);
  });
  await writeBatched(env, statements);
  return statements.length;
}

export interface TruckPrior {
  pVisit: number;
  meanDelta: number;
  sdDelta: number;
  nDays: number;
}

// Shrinkage toward the pooled rate, with the strength set by how much evidence
// the cell actually has: a cell seen on 2 days is ~90% pooled, a cell seen on 40
// is ~83% its own. PRIOR_STRENGTH is the number of days at which a cell's own
// rate and the pooled rate weigh equally — deliberately not tuned, because with
// six weeks of data any tuned value would be fitted to the same nights it would
// then be evaluated on.
const PRIOR_STRENGTH = 8;

export async function loadTruckPriors(env: Env, dow: number): Promise<Map<number, TruckPrior>> {
  const res = await env.DB.prepare(`SELECT dow, hour, p_visit, mean_delta, sd_delta, n_days FROM rebalance_profile`).all<{
    dow: number;
    hour: number;
    p_visit: number;
    mean_delta: number;
    sd_delta: number;
    n_days: number;
  }>();
  const rows = res.results ?? [];
  const out = new Map<number, TruckPrior>();
  if (!rows.length) return out;

  // Pooled over all dow for the same hour — hour-of-day is the strong axis of a
  // truck schedule, weekday the weak one, so pooling across dow loses far less
  // than pooling across hour.
  const pooled = new Map<number, { p: number; mean: number; sd: number; n: number }>();
  for (const r of rows) {
    let p = pooled.get(r.hour);
    if (!p) pooled.set(r.hour, (p = { p: 0, mean: 0, sd: 0, n: 0 }));
    p.p += r.p_visit * r.n_days;
    p.mean += r.mean_delta * r.p_visit * r.n_days;
    p.sd += r.sd_delta * r.n_days;
    p.n += r.n_days;
  }

  for (const r of rows.filter((r) => r.dow === dow)) {
    const pool = pooled.get(r.hour)!;
    const poolP = pool.n ? pool.p / pool.n : 0;
    const poolMean = pool.p > 0 ? pool.mean / pool.p : 0;
    const poolSd = pool.n ? pool.sd / pool.n : 0;
    const w = r.n_days / (r.n_days + PRIOR_STRENGTH);
    out.set(r.hour, {
      pVisit: w * r.p_visit + (1 - w) * poolP,
      meanDelta: w * r.mean_delta + (1 - w) * poolMean,
      // A cell that saw one visit reports sd 0, which would fake certainty about
      // the size of a rare event — floor it at the pooled spread.
      sdDelta: Math.max(w * r.sd_delta + (1 - w) * poolSd, poolSd * 0.5),
      nDays: r.n_days,
    });
  }
  return out;
}
