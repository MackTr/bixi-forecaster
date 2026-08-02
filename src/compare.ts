// The race: four variants, one grading rule, one scoreboard.
//
// The entire value of this service is that `gaussian` and `ml` are scored by
// the SAME code against the SAME definition of the actual on the SAME nights.
// Every asymmetry that creeps in here — a variant that skips a hard night, a
// window measured differently, a start inventory that differs between arms —
// shows up as model skill and is indistinguishable from it.

import type { Env } from "./worker";
import { addDays, minsToHHMM } from "./tz";
import { fetchGaussianPrediction, hhmmToMinutes } from "./monitor";

// `glm` is no longer raced (see docs/model.md — the shadow window runs three
// arms). The value stays in the union and in the schema's CHECK constraint so
// that if it is ever built, `POST /admin/replay` can write its rows into past
// nights from the frozen f_* columns without a migration.
export type Variant = "gaussian" | "ml" | "blend" | "glm";

export interface VariantOutcome {
  variant: Variant;
  probability: number | null;
  predictedMinutes: number | null;
  windowEarly: number | null;
  windowLate: number | null;
  startBikes: number | null;
  modelVersion: string | null;
}

export async function storePrediction(
  env: Env,
  targetDate: string,
  now: number,
  o: VariantOutcome,
  basis: unknown,
  curve: unknown | null,
): Promise<VariantOutcome> {
  // Re-runs update the estimate but never touch actual_/error_/finalized_ —
  // a re-fired cron must not be able to un-grade a night already on the books.
  await env.DB.prepare(
    `INSERT INTO predictions (target_date, variant, created_ts, predicted_minutes, probability,
                              window_early, window_late, start_bikes, model_version, basis_json, curve_json)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(target_date, variant) DO UPDATE SET
       created_ts = excluded.created_ts, predicted_minutes = excluded.predicted_minutes,
       probability = excluded.probability, window_early = excluded.window_early,
       window_late = excluded.window_late, start_bikes = excluded.start_bikes,
       model_version = excluded.model_version, basis_json = excluded.basis_json,
       curve_json = excluded.curve_json`,
  )
    .bind(
      targetDate,
      o.variant,
      now,
      o.predictedMinutes,
      o.probability,
      o.windowEarly,
      o.windowLate,
      o.startBikes,
      o.modelVersion,
      JSON.stringify(basis),
      curve == null ? null : JSON.stringify(curve),
    )
    .run();
  return o;
}

// The control arm, copied verbatim from what bixi-predictor actually published.
// Nothing is recomputed here: if the sibling changes its model tomorrow, this
// mirror changes with it, which is the honest comparison. A missing row is
// recorded as an absent control rather than silently skipped, so the scoreboard
// can report how often the control was unavailable.
export async function mirrorGaussian(env: Env, targetDate: string, now: number): Promise<VariantOutcome | null> {
  const g = await fetchGaussianPrediction(env, targetDate);
  if (!g) return null;
  const o: VariantOutcome = {
    variant: "gaussian",
    probability: g.probability,
    predictedMinutes: g.predicted?.minutes ?? null,
    windowEarly: hhmmToMinutes(g.window?.early),
    windowLate: hhmmToMinutes(g.window?.late),
    startBikes: g.basis?.target?.startBikes ?? null,
    modelVersion: null,
  };
  return storePrediction(env, targetDate, now, o, { mirroredFrom: "bixi-predictor", basis: g.basis ?? null }, null);
}

// ---------------------------------------------------------------------------
// The blend rule. FROZEN BEFORE ANY DATA WAS SEEN, and that is the point.
//
// The tempting move is to fit a blending weight. With ~40 paired nights, a
// weight tuned on those nights and then evaluated on those same nights would
// report a win it has not earned — the blend would beat both parents by
// construction. So the gate is a rule, not a parameter: it reads the Gaussian
// model's OWN published confidence and defers to whichever engine claims to be
// on firmer ground.
//
// bixi-predictor reports fallbackLevel 0 when day-of-week + rain found enough
// similar days, 1 when it had to drop its nudges, 2 when it had to pool
// weekdays with weekends. Levels 1 and 2 are it saying, in its own words, that
// it ran out of precedent — precisely the regime a network-trained model should
// help with. GATE_EFFECTIVE_N sits one above the sibling's own minEffectiveN of
// 3; it was taken from that constant rather than chosen, so that no number here
// is fitted to the nights it will be judged on.
//
// The weights are recorded in basis_json every night, so this rule can be
// revisited honestly later — on data it did not see.
// ---------------------------------------------------------------------------
const GATE_EFFECTIVE_N = 4;

// ---------------------------------------------------------------------------
// Composing two beliefs.
//
// The weights above say how much to trust each arm. They do NOT say how to
// combine what the arms published, and the two questions have different
// answers:
//
//   the TIME   is a point forecast, and averaging two of them cancels
//              independent error — the oldest result in forecast combination,
//              and hard to beat.
//   the WINDOW is an interval that has to contain the truth, and averaging two
//              of those is simply wrong. Gaussian 8:00-8:40 and ML 10:30-11:30
//              averaged at .7/.3 gives 8:45-9:31 — an interval containing
//              NEITHER arm's belief, which fails whichever one was right. The
//              blend would then be punished on window coverage precisely on the
//              nights where blending is doing work.
//
// So the window is read off the MIXTURE distribution: stack the two beliefs
// with the gate's weights, and take the 25th and 75th percentiles of the
// result. That widens honestly when the arms disagree and collapses back to the
// parents' own window when they agree — with no new fitted parameter, which is
// what keeps the rule frozen.
//
// Each arm is reconstructed from the only thing it publishes: (p25, median,
// p75), read as a piecewise-linear CDF. Treating the published window as the
// middle 50% is the same assumption `scoreboard()` already makes when it scores
// windowCoverage, so the blend and its grader agree about what a window means.
// ---------------------------------------------------------------------------

interface CdfPoint {
  x: number;
  f: number;
}

// Tails are mirrored outward from the adjacent inner quartile width. The
// alternative — treating the p25..p75 box as the entire support — would make
// every arm's distribution artificially compact and the mixture's percentiles
// too narrow, reintroducing a milder version of the bug this replaces.
function beliefCdf(early: number, point: number, late: number): CdfPoint[] {
  const p = Math.min(Math.max(point, early), late); // published triples are ordered; do not trust it
  const lo = Math.max(p - early, 1);
  const hi = Math.max(late - p, 1);
  return [
    { x: early - lo, f: 0 },
    { x: early, f: 0.25 },
    { x: p, f: 0.5 },
    { x: late, f: 0.75 },
    { x: late + hi, f: 1 },
  ];
}

function cdfAt(pts: CdfPoint[], x: number): number {
  if (x <= pts[0].x) return 0;
  if (x >= pts[pts.length - 1].x) return 1;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1];
    const b = pts[i];
    if (x <= b.x) return b.x === a.x ? b.f : a.f + ((x - a.x) / (b.x - a.x)) * (b.f - a.f);
  }
  return 1;
}

// Every component is piecewise linear on the union of all breakpoints, so the
// mixture is linear on each segment between them and inverting it exactly is a
// scan plus one interpolation — no root finding, no sampling, deterministic.
export function mixtureQuantile(comps: { w: number; pts: CdfPoint[] }[], target: number): number | null {
  const total = comps.reduce((s, c) => s + c.w, 0);
  if (!(total > 0)) return null;
  const xs = [...new Set(comps.flatMap((c) => c.pts.map((p) => p.x)))].sort((a, b) => a - b);
  const F = (x: number) => comps.reduce((s, c) => s + (c.w / total) * cdfAt(c.pts, x), 0);
  let prevX = xs[0];
  let prevF = F(prevX);
  if (prevF >= target) return prevX;
  for (let i = 1; i < xs.length; i++) {
    const x = xs[i];
    const f = F(x);
    if (f >= target) return f === prevF ? x : prevX + ((target - prevF) / (f - prevF)) * (x - prevX);
    prevX = x;
    prevF = f;
  }
  return xs[xs.length - 1];
}

export function blendWeights(fallbackLevel: number | null, effectiveN: number | null): { gaussian: number; ml: number; reason: string } {
  if (fallbackLevel == null) return { gaussian: 0, ml: 1, reason: "no gaussian basis" };
  if (fallbackLevel === 0 && (effectiveN ?? 0) >= GATE_EFFECTIVE_N) {
    return { gaussian: 0.7, ml: 0.3, reason: `gaussian confident (level 0, nEff>=${GATE_EFFECTIVE_N})` };
  }
  if (fallbackLevel === 0) return { gaussian: 0.5, ml: 0.5, reason: "level 0 but thin nEff" };
  if (fallbackLevel === 1) return { gaussian: 0.4, ml: 0.6, reason: "level 1 — gaussian dropped its nudges" };
  return { gaussian: 0.2, ml: 0.8, reason: "level 2 — gaussian pooled day classes" };
}

export async function predictBlend(env: Env, targetDate: string, now: number): Promise<VariantOutcome | null> {
  const res = await env.DB.prepare(
    `SELECT variant, predicted_minutes, probability, window_early, window_late, start_bikes, basis_json
     FROM predictions WHERE target_date = ? AND variant IN ('gaussian','ml')`,
  )
    .bind(targetDate)
    .all<{
      variant: string;
      predicted_minutes: number | null;
      probability: number | null;
      window_early: number | null;
      window_late: number | null;
      start_bikes: number | null;
      basis_json: string;
    }>();
  const rows = res.results ?? [];
  const g = rows.find((r) => r.variant === "gaussian");
  const m = rows.find((r) => r.variant === "ml");
  if (!g && !m) return null;

  let fallbackLevel: number | null = null;
  let effectiveN: number | null = null;
  if (g) {
    try {
      const b = JSON.parse(g.basis_json) as { basis?: { fallbackLevel?: number; effectiveN?: number } };
      fallbackLevel = b.basis?.fallbackLevel ?? null;
      effectiveN = b.basis?.effectiveN ?? null;
    } catch {
      /* a malformed basis means "no confidence signal", which the gate handles */
    }
  }
  let w = blendWeights(g ? fallbackLevel : null, effectiveN);
  if (!m) w = { gaussian: 1, ml: 0, reason: "ml unavailable" };
  if (!g) w = { gaussian: 0, ml: 1, reason: "gaussian unavailable" };

  // Blend only over the arms that actually produced a number, renormalising —
  // otherwise a variant that abstained would drag the blend toward zero and
  // look like a confident early prediction.
  const mix = (pick: (r: typeof rows[number]) => number | null): number | null => {
    let num = 0;
    let den = 0;
    if (g && w.gaussian > 0) {
      const v = pick(g);
      if (v != null) (num += w.gaussian * v), (den += w.gaussian);
    }
    if (m && w.ml > 0) {
      const v = pick(m);
      if (v != null) (num += w.ml * v), (den += w.ml);
    }
    return den > 0 ? Math.round(num / den) : null;
  };

  // Probabilities are the one quantity that DOES mix linearly — a weighted
  // average of two run-out probabilities is itself a run-out probability.
  const probability = (() => {
    const v = mix((r) => (r.probability == null ? null : r.probability * 1000));
    return v == null ? null : v / 1000;
  })();

  // The mixture, over the arms that published a complete (p25, median, p75).
  // An arm that withheld a window because it thought a run-out unlikely has no
  // shape to contribute and is left out; the weights renormalise, exactly as
  // they do for the point estimate.
  const comps: { w: number; pts: CdfPoint[] }[] = [];
  for (const [row, weight] of [
    [g, w.gaussian],
    [m, w.ml],
  ] as const) {
    if (!row || !(weight > 0)) continue;
    if (row.window_early == null || row.window_late == null || row.predicted_minutes == null) continue;
    comps.push({ w: weight, pts: beliefCdf(row.window_early, row.predicted_minutes, row.window_late) });
  }
  const q = (t: number) => {
    const v = comps.length ? mixtureQuantile(comps, t) : null;
    return v == null ? null : Math.round(v);
  };

  const predictedMinutes = mix((r) => r.predicted_minutes);
  const mixtureMedian = q(0.5);
  const o: VariantOutcome = {
    variant: "blend",
    probability,
    predictedMinutes,
    windowEarly: q(0.25),
    windowLate: q(0.75),
    // Both arms must have been seeded with the same inventory; if they ever
    // disagree the audit in the scoreboard surfaces it rather than this
    // silently preferring one.
    startBikes: m?.start_bikes ?? g?.start_bikes ?? null,
    modelVersion: null,
  };
  return storePrediction(
    env,
    targetDate,
    now,
    o,
    {
      rule: "frozen-gate-v2-mixture",
      weights: w,
      fallbackLevel,
      effectiveN,
      // The alternative point estimate, recorded nightly and graded by nobody.
      // Where the weighted mean lands between two far-apart arms, the mixture
      // median instead sits inside whichever arm holds the majority of the
      // weight — a time one model actually believes. Which is better is an
      // empirical question this window is too small to settle, so it is stored
      // rather than chosen, and can be evaluated later on data it did not see.
      // Same discipline as the gate itself.
      mixtureMedian,
      pointRule: "weighted-mean",
      components: comps.length,
    },
    null,
  );
}

// ---------------------------------------------------------------------------
// Scoreboard
// ---------------------------------------------------------------------------

export interface VariantScore {
  variant: string;
  n: number; // graded nights
  mae: number | null; // over nights where both a prediction and an actual exist
  bias: number | null; // mean signed error — late-vs-early, which MAE hides
  windowCoverage: number | null; // share of actuals landing inside P25-P75
  brier: number | null; // on the run-out probability
  missingNights: number; // nights this variant produced nothing
}

interface ScoreRow {
  variant: string;
  target_date: string;
  predicted_minutes: number | null;
  probability: number | null;
  window_early: number | null;
  window_late: number | null;
  start_bikes: number | null;
  actual_minutes: number | null;
  finalized_ts: number | null;
}

export async function scoreboard(env: Env, days: number, today: string) {
  const from = addDays(today, -days);
  const res = await env.DB.prepare(
    `SELECT variant, target_date, predicted_minutes, probability, window_early, window_late,
            start_bikes, actual_minutes, finalized_ts
     FROM predictions WHERE target_date >= ? AND target_date <= ? AND finalized_ts IS NOT NULL
     ORDER BY target_date ASC`,
  )
    .bind(from, today)
    .all<ScoreRow>();
  const rows = res.results ?? [];
  const dates = [...new Set(rows.map((r) => r.target_date))];
  const byVariant = new Map<string, ScoreRow[]>();
  for (const r of rows) {
    if (!byVariant.has(r.variant)) byVariant.set(r.variant, []);
    byVariant.get(r.variant)!.push(r);
  }

  const scores: VariantScore[] = [];
  for (const [variant, vr] of byVariant) {
    const graded = vr.filter((r) => r.predicted_minutes != null && r.actual_minutes != null);
    const errs = graded.map((r) => r.predicted_minutes! - r.actual_minutes!);
    const covered = graded.filter(
      (r) => r.window_early != null && r.window_late != null && r.actual_minutes! >= r.window_early && r.actual_minutes! <= r.window_late,
    );
    // Brier over every finalized night, including the ones nothing ran out on —
    // scoring only the run-out nights would reward a model that cries wolf.
    const probRows = vr.filter((r) => r.probability != null);
    const brier = probRows.length
      ? probRows.reduce((s, r) => s + Math.pow(r.probability! - (r.actual_minutes != null ? 1 : 0), 2), 0) / probRows.length
      : null;
    scores.push({
      variant,
      n: graded.length,
      mae: errs.length ? Math.round((errs.reduce((s, e) => s + Math.abs(e), 0) / errs.length) * 10) / 10 : null,
      bias: errs.length ? Math.round((errs.reduce((s, e) => s + e, 0) / errs.length) * 10) / 10 : null,
      windowCoverage: graded.length ? Math.round((covered.length / graded.length) * 1000) / 1000 : null,
      brier: brier == null ? null : Math.round(brier * 1000) / 1000,
      missingNights: dates.length - vr.length,
    });
  }

  // Paired comparisons: only nights where BOTH arms produced a number count.
  // An unpaired MAE would let a model look good by abstaining on hard nights.
  const pairs: PairedResult[] = [];
  const names = [...byVariant.keys()];
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      pairs.push(pairedCompare(byVariant.get(names[i])!, byVariant.get(names[j])!, names[i], names[j]));
    }
  }

  // A fairness audit, not a statistic: every arm must have been handed the same
  // 10pm inventory. A non-empty list here invalidates the comparison outright,
  // so it is reported alongside the numbers rather than buried.
  const seedMismatches: string[] = [];
  for (const d of dates) {
    const seeds = new Set(rows.filter((r) => r.target_date === d && r.start_bikes != null).map((r) => r.start_bikes));
    if (seeds.size > 1) seedMismatches.push(d);
  }

  return {
    window: { from, to: today, days },
    gradedNights: dates.length,
    scores: scores.sort((a, b) => (a.mae ?? 1e9) - (b.mae ?? 1e9)),
    paired: pairs,
    seedMismatches,
    // Published with every reading, deliberately. At n≈40 paired nights and the
    // ~56-minute residual spread this problem has, only a difference of roughly
    // 20-25 minutes in MAE is distinguishable from noise. Without this line
    // somebody reads ten good nights and declares victory.
    interpretation: {
      detectableEffectMinutes: 22,
      note:
        "Paired MAE differences smaller than ~20-25 min at n≈40 are not distinguishable from noise. " +
        "Require the sign test as well: a real winner wins MORE NIGHTS, not just a better average, " +
        "which is robust to the one catastrophic night that can swing an MAE on its own.",
    },
  };
}

export interface PairedResult {
  a: string;
  b: string;
  n: number;
  maeA: number | null;
  maeB: number | null;
  meanDiff: number | null; // maeA - maeB; negative = a is better
  aWins: number;
  bWins: number;
  ties: number;
  signTestP: number | null; // two-sided
}

function pairedCompare(ra: ScoreRow[], rb: ScoreRow[], a: string, b: string): PairedResult {
  const mapB = new Map(rb.map((r) => [r.target_date, r]));
  let sa = 0;
  let sb = 0;
  let n = 0;
  let aWins = 0;
  let bWins = 0;
  let ties = 0;
  for (const x of ra) {
    const y = mapB.get(x.target_date);
    if (!y || x.predicted_minutes == null || y.predicted_minutes == null || x.actual_minutes == null) continue;
    // Both errors are measured against x's actual, not each row's own. They are
    // the same night so the two values should be identical — scoring each arm
    // against its own copy would let a stale finalize on one variant hand it a
    // different truth than its opponent.
    const ea = Math.abs(x.predicted_minutes - x.actual_minutes);
    const eb = Math.abs(y.predicted_minutes - x.actual_minutes);
    sa += ea;
    sb += eb;
    n++;
    if (ea < eb) aWins++;
    else if (eb < ea) bWins++;
    else ties++;
  }
  return {
    a,
    b,
    n,
    maeA: n ? Math.round((sa / n) * 10) / 10 : null,
    maeB: n ? Math.round((sb / n) * 10) / 10 : null,
    meanDiff: n ? Math.round(((sa - sb) / n) * 10) / 10 : null,
    aWins,
    bWins,
    ties,
    signTestP: aWins + bWins > 0 ? signTestTwoSided(aWins, aWins + bWins) : null,
  };
}

// Exact two-sided binomial test at p=0.5. Ties are dropped, which is the
// standard sign test convention. Exact rather than normal-approximated because
// n here is ~40 and the approximation is poor in the tail that matters.
function signTestTwoSided(k: number, n: number): number {
  const logC = (n: number, k: number) => lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
  const pmf = (i: number) => Math.exp(logC(n, i) - n * Math.LN2);
  const observed = pmf(k);
  let p = 0;
  // Sum every outcome at most as likely as the observed one — the standard
  // definition, and it handles the asymmetry ties introduce.
  for (let i = 0; i <= n; i++) if (pmf(i) <= observed * (1 + 1e-9)) p += pmf(i);
  return Math.min(1, Math.round(p * 10000) / 10000);
}

// Lanczos approximation — accurate well past the n≈40 this is used at.
function lgamma(z: number): number {
  const g = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < g.length; i++) x += g[i] / (z + i + 1);
  const t = z + g.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

export const fmtMinutes = minsToHHMM;
