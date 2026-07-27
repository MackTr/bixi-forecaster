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

  const probability = (() => {
    const v = mix((r) => (r.probability == null ? null : r.probability * 1000));
    return v == null ? null : v / 1000;
  })();
  const o: VariantOutcome = {
    variant: "blend",
    probability,
    predictedMinutes: mix((r) => r.predicted_minutes),
    windowEarly: mix((r) => r.window_early),
    windowLate: mix((r) => r.window_late),
    // Both arms must have been seeded with the same inventory; if they ever
    // disagree the audit in the scoreboard surfaces it rather than this
    // silently preferring one.
    startBikes: m?.start_bikes ?? g?.start_bikes ?? null,
    modelVersion: null,
  };
  return storePrediction(env, targetDate, now, o, { rule: "frozen-gate-v1", weights: w, fallbackLevel, effectiveN }, null);
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
