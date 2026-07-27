// Nightly orchestration.
//
// CRITICAL DATE RULE, inherited from bixi-predictor and just as load-bearing
// here: the cron fires at 02:05/03:05 UTC, when the UTC calendar date is
// already *tomorrow* in Montreal terms. Every date below must flow from
// localToday()/addDays() — never from toISOString() or any UTC-derived Date
// field.
//
// Every step is individually try/caught, so a monitor outage at 10pm still
// yields predictions from existing facts, and a missed night self-heals on the
// next one (14-day sync lookback, actuals re-fetch, finalization of ALL
// unfinalized past predictions across ALL variants).

import type { Env } from "./worker";
import { addDays, localToday } from "./tz";
import { datesNeedingSync, syncDays } from "./sync";
import { datesNeedingActuals, fetchActualsFor, fetchForecastFor, freezeForecast, upsertActuals } from "./weather";
import { buildRebalanceProfile } from "./rebalance";
import { predictMl, predictGlm } from "./demand";
import { mirrorGaussian, predictBlend, type VariantOutcome } from "./compare";

export interface PipelineResult {
  today: string;
  targetDate: string;
  synced: string[];
  hourlyRows: number;
  weatherActuals: number;
  forecastFrozen: number;
  rebalanceCells: number;
  finalized: string[];
  variants: Record<string, VariantOutcome | null>;
  errors: string[];
}

// Fill actual_minutes/error_minutes for every past prediction whose target day
// now has a complete daily_facts row — for EVERY variant, in one statement, so
// no arm of the comparison can be graded on a different rule or a different
// day's data than another. Not just yesterday's: this self-heals nights the
// cron missed. Today's own prediction is also graded as soon as a run-out is on
// the books (the day's FIRST transition can't change once it happened). A
// no-run-out day still waits for complete=1: "never ran out" isn't known until
// midnight.
async function finalizePastPredictions(env: Env, today: string, now: number): Promise<string[]> {
  const res = await env.DB.prepare(
    `UPDATE predictions SET
       actual_minutes = (SELECT f.runout_minutes FROM daily_facts f WHERE f.date = predictions.target_date),
       error_minutes = CASE WHEN predicted_minutes IS NOT NULL
         THEN predicted_minutes - (SELECT f.runout_minutes FROM daily_facts f WHERE f.date = predictions.target_date)
         END,
       finalized_ts = ?
     WHERE finalized_ts IS NULL
       AND EXISTS (
         SELECT 1 FROM daily_facts f WHERE f.date = predictions.target_date
           AND ((predictions.target_date < ? AND f.complete = 1)
                OR (predictions.target_date = ? AND f.runout_minutes IS NOT NULL))
       )
     RETURNING target_date`,
  )
    .bind(now, today, today)
    .all<{ target_date: string }>();
  return [...new Set((res.results ?? []).map((r) => r.target_date))];
}

export async function runNightly(env: Env, opts: { now?: number }): Promise<PipelineResult> {
  const now = opts.now ?? Math.floor(Date.now() / 1000);
  const today = localToday(now);
  const targetDate = addDays(today, 1);
  const result: PipelineResult = {
    today,
    targetDate,
    synced: [],
    hourlyRows: 0,
    weatherActuals: 0,
    forecastFrozen: 0,
    rebalanceCells: 0,
    finalized: [],
    variants: { ml: null, glm: null, gaussian: null, blend: null },
    errors: [],
  };
  const step = async (name: string, fn: () => Promise<void>) => {
    try {
      await fn();
    } catch (e) {
      result.errors.push(`${name}: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  await step("sync-hourly", async () => {
    const need = await datesNeedingSync(env, 14, today);
    const days = await syncDays(env, need, now);
    result.synced = days.map((d) => d.date);
    result.hourlyRows = days.reduce((s, d) => s + d.hours, 0);
  });
  await step("weather-actuals", async () => {
    const need = await datesNeedingActuals(env, today);
    if (!need.length) return;
    result.weatherActuals = await upsertActuals(env, await fetchActualsFor(env, need, today), now);
  });
  // Frozen before anything reads it, and never rewritten afterwards.
  await step("weather-forecast", async () => {
    result.forecastFrozen = await freezeForecast(env, await fetchForecastFor(env, targetDate), now);
  });
  await step("rebalance-profile", async () => {
    result.rebalanceCells = await buildRebalanceProfile(env, today, now);
  });
  await step("finalize", async () => {
    result.finalized = await finalizePastPredictions(env, today, now);
  });

  // The four arms. Each is its own step: a failure in one must not deny the
  // others a row, or the paired comparison silently loses that night for
  // everybody — which biases the scoreboard toward whichever model fails least
  // often rather than whichever predicts best.
  await step("predict-ml", async () => {
    result.variants.ml = await predictMl(env, targetDate, now);
  });
  await step("predict-glm", async () => {
    result.variants.glm = await predictGlm(env, targetDate, now);
  });
  await step("mirror-gaussian", async () => {
    result.variants.gaussian = await mirrorGaussian(env, targetDate, now);
  });
  // Last: the blend gate reads the two rows above.
  await step("predict-blend", async () => {
    result.variants.blend = await predictBlend(env, targetDate, now);
  });

  return result;
}

// First-deploy / repair path: digest N days of history + weather actuals without
// forecasting or predicting. `force` re-syncs days already marked complete —
// needed when a new derived column must be filled for existing history.
export async function backfill(env: Env, days: number, opts?: { force?: boolean; now?: number }): Promise<PipelineResult> {
  const now = opts?.now ?? Math.floor(Date.now() / 1000);
  const today = localToday(now);
  const result: PipelineResult = {
    today,
    targetDate: addDays(today, 1),
    synced: [],
    hourlyRows: 0,
    weatherActuals: 0,
    forecastFrozen: 0,
    rebalanceCells: 0,
    finalized: [],
    variants: {},
    errors: [],
  };
  const step = async (name: string, fn: () => Promise<void>) => {
    try {
      await fn();
    } catch (e) {
      result.errors.push(`${name}: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  await step("sync-hourly", async () => {
    let need: string[];
    if (opts?.force) {
      need = [];
      for (let d = addDays(today, -days); d <= today; d = addDays(d, 1)) need.push(d);
    } else {
      need = await datesNeedingSync(env, days, today);
    }
    const synced = await syncDays(env, need, now);
    result.synced = synced.map((d) => d.date);
    result.hourlyRows = synced.reduce((s, d) => s + d.hours, 0);
  });
  await step("weather-actuals", async () => {
    const need = await datesNeedingActuals(env, today);
    if (!need.length) return;
    result.weatherActuals = await upsertActuals(env, await fetchActualsFor(env, need, today), now);
  });
  await step("rebalance-profile", async () => {
    result.rebalanceCells = await buildRebalanceProfile(env, today, now);
  });
  await step("finalize", async () => {
    result.finalized = await finalizePastPredictions(env, today, now);
  });
  return result;
}
