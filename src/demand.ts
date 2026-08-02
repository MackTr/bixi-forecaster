// Stage 1 plus the join to stage 2: turn an artifact into tonight's prediction.
//
// The drift problem this file exists to solve: station 345 grows about +14% a
// year, and BIXI's open data lags a quarter, so the newest trips available for
// training are already stale by the time they are used. A tree handed a raw
// `year` feature learns a step function and then extrapolates it FLAT — the one
// shape guaranteed to be wrong.
//
// So the trees never see a level at all. They predict a normalized demand
// SHAPE, and a single scalar is refit every night against the last 28 observed
// mornings. Extrapolating a trend is the one thing trees cannot do, and it is
// the only thing kept outside them.

import type { Env } from "./worker";
import { addDays, dowOf } from "./tz";
import { loadActiveArtifact, type Artifact } from "./artifact";
import { assertFeatureContract, encode, type RawHour } from "./features";
import { predict as gbdtPredict, predictLinear } from "./gbdt";
import { loadTruckPriors } from "./rebalance";
import { simulate, seedFor } from "./simulate";
import { storePrediction, type VariantOutcome } from "./compare";

// The morning window everything is normalized against — the same 06:00-11:00
// the 402k-row measurement used, so the offline level and this one mean the
// same thing.
const MORNING_HOURS = [6, 7, 8, 9, 10];
const CALIBRATION_DAYS = 28;
// Below this the level is too poorly determined to trust; fall back to the
// artifact's own training-time level rather than chase noise.
const MIN_CALIBRATION_POINTS = 12;

interface WeatherRow {
  date: string;
  hour: number;
  temp_c: number | null;
  precip_mm: number | null;
  precip_prob: number | null;
  wind_kmh: number | null;
  dew_c: number | null;
}

// `frozen` selects the f_* columns — the forecast as it stood at 10pm — and is
// what every prediction and every replay uses. Calibration passes false to read
// a_* actuals, which is correct there: it is fitting a level to what actually
// happened, not pretending to have known it in advance.
async function loadWeather(env: Env, from: string, to: string, frozen: boolean): Promise<Map<string, WeatherRow>> {
  const cols = frozen
    ? "f_temp_c AS temp_c, f_precip_mm AS precip_mm, f_precip_prob AS precip_prob, f_wind_kmh AS wind_kmh, f_dew_c AS dew_c"
    : "a_temp_c AS temp_c, a_precip_mm AS precip_mm, NULL AS precip_prob, a_wind_kmh AS wind_kmh, a_dew_c AS dew_c";
  const res = await env.DB.prepare(`SELECT date, hour, ${cols} FROM weather_hourly WHERE date >= ? AND date <= ?`)
    .bind(from, to)
    .all<WeatherRow>();
  const m = new Map<string, WeatherRow>();
  for (const r of res.results ?? []) m.set(`${r.date}|${r.hour}`, r);
  return m;
}

// Hourly precipitation at label H covers [H-1, H), so the 20:00-02:00 night is
// labels 21..23 of the evening plus 00..02 of the morning. Identical to
// bixi-predictor's window, and to the offline feature builder's.
function nightPrecip(wx: Map<string, WeatherRow>, date: string): number | null {
  const prev = addDays(date, -1);
  let sum: number | null = null;
  for (const [d, h] of [
    [prev, 21],
    [prev, 22],
    [prev, 23],
    [date, 0],
    [date, 1],
    [date, 2],
  ] as [string, number][]) {
    const v = wx.get(`${d}|${h}`)?.precip_mm;
    if (v != null) sum = (sum ?? 0) + v;
  }
  return sum;
}

function rawHour(wx: Map<string, WeatherRow>, date: string, hour: number, night: number | null): RawHour {
  const w = wx.get(`${date}|${hour}`);
  return {
    date,
    hour,
    tempC: w?.temp_c ?? null,
    precipMm: w?.precip_mm ?? null,
    precipProb: w?.precip_prob ?? null,
    windKmh: w?.wind_kmh ?? null,
    dewC: w?.dew_c ?? null,
    nightPrecipMm: night,
  };
}

type Scorer = (x: number[], which: "departures" | "arrivals") => number;

function scorerFor(a: Artifact, kind: "gbdt" | "glm"): Scorer {
  if (kind === "glm") {
    if (!a.glm) throw new Error(`artifact ${a.version} has no glm block`);
    const g = a.glm;
    return (x, which) => predictLinear(g[which].intercept, g[which].coef, x);
  }
  return (x, which) => gbdtPredict(a[which], x);
}

export interface LevelFit {
  level: number;
  points: number;
  source: "calibrated" | "artifact";
  // Ratio to the training-time level. Published because it IS the drift
  // measurement: a value creeping up over months is the +14%/year trend showing
  // itself, and a sudden jump means something broke rather than grew.
  ratioToTraining: number;
}

// Least squares through the origin: level = sum(shape*obs) / sum(shape^2).
// Only departures the station could actually serve are counted — an hour that
// spent time empty recorded fewer departures than were wanted, and fitting to
// censored counts would drag the level down exactly on the busiest days, which
// are the ones this whole service is about.
export async function calibrateLevel(env: Env, today: string, a: Artifact, score: Scorer): Promise<LevelFit> {
  const from = addDays(today, -CALIBRATION_DAYS);
  const trainingLevel = Math.exp(a.station.logLevel);
  const res = await env.DB.prepare(
    `SELECT h.date, h.hour, h.organic_out, h.empty_minutes
     FROM hourly_facts h JOIN daily_facts f ON f.date = h.date
     WHERE h.date >= ? AND h.date < ? AND f.complete = 1 AND h.obs_count > 0
       AND h.hour BETWEEN ? AND ?`,
  )
    .bind(from, today, MORNING_HOURS[0], MORNING_HOURS[MORNING_HOURS.length - 1])
    .all<{ date: string; hour: number; organic_out: number; empty_minutes: number }>();
  const rows = (res.results ?? []).filter((r) => r.empty_minutes <= 15);
  if (rows.length < MIN_CALIBRATION_POINTS) {
    return { level: trainingLevel, points: rows.length, source: "artifact", ratioToTraining: 1 };
  }

  const wx = await loadWeather(env, addDays(from, -1), today, false);
  const nightCache = new Map<string, number | null>();
  let num = 0;
  let den = 0;
  let used = 0;
  for (const r of rows) {
    if (!nightCache.has(r.date)) nightCache.set(r.date, nightPrecip(wx, r.date));
    const shape = score(encode(rawHour(wx, r.date, r.hour, nightCache.get(r.date)!), a), "departures");
    if (!(shape > 0)) continue;
    num += shape * r.organic_out;
    den += shape * shape;
    used++;
  }
  if (!(den > 0) || used < MIN_CALIBRATION_POINTS) {
    return { level: trainingLevel, points: used, source: "artifact", ratioToTraining: 1 };
  }
  const level = num / den;
  return {
    level,
    points: used,
    source: "calibrated",
    ratioToTraining: Math.round((level / trainingLevel) * 1000) / 1000,
  };
}

async function predictVariant(
  env: Env,
  targetDate: string,
  now: number,
  kind: "gbdt" | "glm",
  variant: "ml" | "glm",
): Promise<VariantOutcome | null> {
  const artifact = await loadActiveArtifact(env, "gbdt");
  if (!artifact) return null; // no activated model yet — shadow mode has not begun
  assertFeatureContract(artifact);
  const score = scorerFor(artifact, kind);

  const today = addDays(targetDate, -1);
  const wx = await loadWeather(env, today, targetDate, true);
  const night = nightPrecip(wx, targetDate);
  const fit = await calibrateLevel(env, today, artifact, score);

  const lambdaDep = new Array(24).fill(0);
  const lambdaArr = new Array(24).fill(0);
  for (let h = 0; h < 24; h++) {
    const x = encode(rawHour(wx, targetDate, h, night), artifact);
    lambdaDep[h] = score(x, "departures") * fit.level;
    lambdaArr[h] = score(x, "arrivals") * fit.level;
  }
  // The evening hours the simulation starts in belong to TONIGHT, not to the
  // target date — using the target date's own hour-22 rate would apply
  // tomorrow's weather to tonight's rack.
  const tonightNight = nightPrecip(wx, today);
  for (const h of [22, 23]) {
    const x = encode(rawHour(wx, today, h, tonightNight), artifact);
    lambdaDep[h] = score(x, "departures") * fit.level;
    lambdaArr[h] = score(x, "arrivals") * fit.level;
  }

  const seed = await env.DB.prepare(`SELECT evening_bikes FROM daily_facts WHERE date = ?`)
    .bind(today)
    .first<{ evening_bikes: number | null }>();
  const startBikes = seed?.evening_bikes ?? null;
  if (startBikes == null) return null; // no 10pm observation = nothing honest to seed with

  const trucks = await loadTruckPriors(env, dowOf(targetDate));
  const sim = simulate({
    startBikes,
    capacity: parseInt(env.STATION_CAPACITY, 10) || 19,
    lambdaDep,
    lambdaArr,
    trucks,
    seed: seedFor(targetDate, variant === "glm" ? 1 : 0),
  });

  // Same publication contract as bixi-predictor, so the monitor dashboard reads
  // either service unchanged: below the probability floor the service says
  // "unlikely" and withholds a time rather than publishing a median nobody
  // should act on.
  const unlikely = sim.probability < 0.5;
  const o: VariantOutcome = {
    variant,
    probability: Math.round(sim.probability * 1000) / 1000,
    predictedMinutes: unlikely ? null : sim.medianMinutes,
    windowEarly: unlikely ? null : sim.p25,
    windowLate: unlikely ? null : sim.p75,
    startBikes,
    modelVersion: artifact.version,
  };
  return storePrediction(
    env,
    targetDate,
    now,
    o,
    {
      engine: kind,
      modelVersion: artifact.version,
      level: fit,
      medianEvenIfUnlikely: unlikely ? sim.medianMinutes : null,
      paths: sim.paths,
      seed: seedFor(targetDate, variant === "glm" ? 1 : 0),
      lambdaDep: lambdaDep.map((v) => Math.round(v * 1000) / 1000),
      lambdaArr: lambdaArr.map((v) => Math.round(v * 1000) / 1000),
      truckHours: [...trucks.entries()].map(([h, t]) => ({ hour: h, pVisit: Math.round(t.pVisit * 100) / 100, nDays: t.nDays })),
    },
    sim.curve,
  );
}

export const predictMl = (env: Env, targetDate: string, now: number) => predictVariant(env, targetDate, now, "gbdt", "ml");

// NOT wired into the nightly pipeline — the shadow window races three arms, and
// src/glm.ts was never written, so this throws `artifact ... has no glm block`.
// Kept because the plumbing it needs (scorerFor, predictLinear, the artifact's
// optional glm block) is already here and correct: building glm.ts and adding
// one pipeline step is all that reviving the fourth arm would take.
export const predictGlm = (env: Env, targetDate: string, now: number) => predictVariant(env, targetDate, now, "glm", "glm");
