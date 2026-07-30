// Stage 2: turn an hourly demand curve into a run-out time.
//
// Stage 1 (the ML model) can only ever learn DEMAND, because BIXI's open trip
// data has no inventory column. Demand is not a run-out time — the same 14
// departures empty a rack that started at 9 and leave one that started at 19
// half full. So the two stages are separate, and the join between them is the
// one number the model does not have to guess: tonight's ACTUAL 10pm count,
// observed by bixi-monitor an hour ago.
//
// That seeding is also what makes the thin truck prior survivable. The largest
// rebalancing event of the day is the evening sweep, and it is already inside
// the initial condition; only overnight and morning moves have to be modelled.
//
// Monte Carlo rather than a closed form because the answer wanted is a
// distribution, not a mean. Poisson arrivals against a hard floor at zero and a
// hard ceiling at capacity is a censored, path-dependent process; the P25-P75
// window falls out of simulating it and would have to be faked analytically.

import type { TruckPrior } from "./rebalance";

export interface SimInput {
  startBikes: number;
  capacity: number;
  // Departures and arrivals per hour, indexed by local hour 0..23. Hours before
  // the 22:00 start and after noon are ignored.
  lambdaDep: number[];
  lambdaArr: number[];
  trucks: Map<number, TruckPrior>; // by local hour, for the TARGET day's dow
  paths?: number;
  seed: number;
}

export interface SimResult {
  probability: number; // share of paths that hit zero before noon
  medianMinutes: number | null; // median over paths that DID run out
  p25: number | null;
  p75: number | null;
  paths: number;
  // Mean simulated inventory at each 15-minute step, for the curve the API
  // publishes — the explainability that makes a wrong answer diagnosable.
  curve: { minutes: number; meanBikes: number }[];
}

// 22:00 through 12:00 next day, in quarter hours. Minutes are expressed since
// the TARGET day's local midnight, so the evening hours are negative and the
// published run-out time needs no adjustment.
const START_MIN = -120; // 22:00 the previous evening
const END_MIN = 12 * 60; // noon
const STEP_MIN = 15;
const STEPS = (END_MIN - START_MIN) / STEP_MIN;
const DEFAULT_PATHS = 150;

// mulberry32 — small, fast, and above all SEEDED. A fixed seed per target date
// is what makes /admin/replay reproduce a night exactly, which is the only way
// to prove a change altered the model rather than the dice.
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Knuth's method. Exact for the small rates a 15-minute bucket produces at a
// 19-dock station (lambda is typically under 5), and it needs no tables.
function poisson(rand: () => number, lambda: number): number {
  if (!(lambda > 0)) return 0;
  if (lambda > 30) return Math.round(lambda + Math.sqrt(lambda) * gaussian(rand)); // guard, not a hot path
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= rand();
  } while (p > L);
  return k - 1;
}

// Box-Muller, one value per call. Truck deltas are the only Gaussian draw here.
function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

const hourOf = (minutes: number) => ((Math.floor(minutes / 60) % 24) + 24) % 24;

export function simulate(input: SimInput): SimResult {
  const paths = input.paths ?? DEFAULT_PATHS;
  const rand = rng(input.seed);
  const cap = Math.max(1, input.capacity);
  const runouts: number[] = [];
  const sumBikes = new Float64Array(STEPS + 1);

  for (let p = 0; p < paths; p++) {
    let bikes = Math.max(0, Math.min(cap, input.startBikes));
    let ranOut: number | null = null;
    sumBikes[0] += bikes;

    for (let s = 0; s < STEPS; s++) {
      const minutes = START_MIN + s * STEP_MIN;
      const h = hourOf(minutes);

      // Truck visits are resolved at hour boundaries, which is roughly how they
      // occur: a van shows up, works the rack, leaves. Spreading the same
      // volume across four quarter-hours would understate how abruptly it moves
      // the inventory, and abruptness is exactly what decides a run-out.
      if (minutes % 60 === 0) {
        const t = input.trucks.get(h);
        if (t && rand() < t.pVisit) {
          const delta = Math.round(t.meanDelta + t.sdDelta * gaussian(rand));
          bikes = Math.max(0, Math.min(cap, bikes + delta));
        }
      }

      // Arrivals first, then departures, within a step. The order matters only
      // when the rack is at a boundary, and this ordering is the conservative
      // one for the question being asked: a bike returned in the same quarter
      // hour someone wants one is available to them, so the model does not
      // invent run-outs that a real cyclist would not have experienced.
      const arr = poisson(rand, Math.max(0, input.lambdaArr[h] ?? 0) / 4);
      bikes = Math.min(cap, bikes + arr);
      const want = poisson(rand, Math.max(0, input.lambdaDep[h] ?? 0) / 4);
      const stock = bikes; // before departures — needed to place the run-out inside the step
      const took = Math.min(want, bikes);
      bikes -= took;

      // Censoring, made explicit: `want - took` is demand that walked away. The
      // trip data this model trains on never recorded those riders either,
      // which is why the demand curve is systematically light in exactly the
      // hours around a run-out. Documented in docs/model.md rather than
      // silently corrected, because correcting it would mean inventing riders.
      if (bikes <= 0 && ranOut == null) {
        // Interpolate within the step instead of snapping to its boundary: at
        // 15-minute granularity, always reporting the end of the bucket would
        // add a systematic ~7 minute late bias to every prediction.
        //
        // The fraction is stock/want, not anything involving `took`. `took` is
        // Math.min(want, stock), so once the rack empties it EQUALS stock and any
        // ratio of the two collapses to 1 — which silently reinstated exactly the
        // end-of-bucket bias this interpolation exists to remove. Spreading `want`
        // uniformly across the step, the rack hits zero after stock/want of it.
        const frac = want > 0 ? Math.min(1, stock / want) : 1;
        ranOut = minutes + STEP_MIN * frac;
      }
      sumBikes[s + 1] += bikes;
    }
    if (ranOut != null) runouts.push(ranOut);
  }

  const probability = runouts.length / paths;
  runouts.sort((a, b) => a - b);
  const q = (f: number): number | null => {
    if (!runouts.length) return null;
    return Math.round(runouts[Math.min(runouts.length - 1, Math.floor(f * runouts.length))]);
  };

  const curve: { minutes: number; meanBikes: number }[] = [];
  for (let s = 0; s <= STEPS; s++) {
    curve.push({
      minutes: START_MIN + s * STEP_MIN,
      meanBikes: Math.round((sumBikes[s] / paths) * 100) / 100,
    });
  }

  return { probability, medianMinutes: q(0.5), p25: q(0.25), p75: q(0.75), paths, curve };
}

// A stable per-date seed: same date, same dice, every replay. Deliberately NOT
// derived from the clock — a time-seeded run could never be reproduced, and
// reproducing a night is the whole point of the replay endpoint.
export function seedFor(targetDate: string, salt = 0): number {
  let h = 2166136261 >>> 0;
  const s = `${targetDate}#${salt}`;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}
