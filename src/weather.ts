// Hourly weather from Open-Meteo (free, no key), stored with the forecast and
// the actual SIDE BY SIDE.
//
// bixi-predictor overwrites each forecast with the actual once a day settles.
// That is fine for a model that only ever reasons forward, and fatal for a
// service whose whole purpose is to be graded: once the forecast is gone you
// can no longer replay a night on the inputs the model actually had, and you
// can never measure how wrong the forecasts were. So here f_* is written ONCE
// (guarded in SQL, not by convention) and a_* accumulates beside it.
//
// Sources, unchanged from the sibling because the reasoning still holds:
//  - forecast API: tomorrow's forecast AND recent-past actuals (it serves past
//    dates ~92 days back with no delay)
//  - archive API (ERA5): deep backfill only — it lags realtime by ~5 days and
//    would silently return nulls for the most important (recent) days, and it
//    has no precipitation_probability at all.
//
// CONVENTION, and it must match training exactly: with timezone=America/Toronto
// the hourly labels are local wall-clock, and `precipitation` at label H is the
// sum over the PRECEDING hour [H-1, H). The offline panel builder applies the
// same rule; a mismatch here would shift rain by one hour between train and
// serve and would not raise a single error.

import type { Env } from "./worker";
import { TZ, addDays, dayDiff } from "./tz";
import { writeBatched } from "./sync";

const FORECAST_URL = "https://api.open-meteo.com/v1/forecast";
const ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive";
const RECENT_DAYS = 7; // age up to which the forecast API is the actuals source

export interface HourWeather {
  date: string;
  hour: number;
  tempC: number | null;
  precipMm: number | null;
  precipProb: number | null; // forecast only
  windKmh: number | null;
  dewC: number | null;
}

interface OpenMeteoHourly {
  time: string[]; // "YYYY-MM-DDTHH:MM" local labels
  temperature_2m?: (number | null)[];
  precipitation?: (number | null)[];
  precipitation_probability?: (number | null)[];
  wind_speed_10m?: (number | null)[];
  dew_point_2m?: (number | null)[];
}

async function fetchHourly(
  base: string,
  env: Env,
  startDate: string,
  endDate: string,
  withProb: boolean,
): Promise<HourWeather[]> {
  const vars = ["temperature_2m", "precipitation", "wind_speed_10m", "dew_point_2m"];
  if (withProb) vars.push("precipitation_probability");
  const u = new URL(base);
  u.searchParams.set("latitude", env.STATION_LAT);
  u.searchParams.set("longitude", env.STATION_LON);
  u.searchParams.set("hourly", vars.join(","));
  u.searchParams.set("timezone", TZ);
  u.searchParams.set("start_date", startDate);
  u.searchParams.set("end_date", endDate);
  const res = await fetch(u.toString(), { headers: { "User-Agent": "bixi-forecaster (personal)" } });
  if (!res.ok) throw new Error(`open-meteo ${res.status} for ${startDate}..${endDate}`);
  const body = (await res.json()) as { hourly?: OpenMeteoHourly };
  const h = body.hourly;
  if (!h?.time) throw new Error("open-meteo: no hourly data");
  return h.time.map((label, i) => {
    const [date, hhmm] = label.split("T");
    return {
      date,
      hour: parseInt(hhmm.slice(0, 2), 10),
      tempC: h.temperature_2m?.[i] ?? null,
      precipMm: h.precipitation?.[i] ?? null,
      precipProb: h.precipitation_probability?.[i] ?? null,
      windKmh: h.wind_speed_10m?.[i] ?? null,
      dewC: h.dew_point_2m?.[i] ?? null,
    };
  });
}

// Freeze a forecast. The WHERE clause is the guarantee: an existing frozen row
// is left exactly as it was, so re-running the pipeline (or a repair run days
// later, when "the forecast" would mean something entirely different) cannot
// rewrite history. This is enforced in SQL rather than by a caller-side check
// because every future caller would have to remember, and one that forgets
// produces a silently optimistic backtest.
export async function freezeForecast(env: Env, hours: HourWeather[], now: number): Promise<number> {
  const stmt = env.DB.prepare(
    `INSERT INTO weather_hourly (date, hour, f_temp_c, f_precip_mm, f_precip_prob, f_wind_kmh, f_dew_c, f_fetched_ts)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(date, hour) DO UPDATE SET
       f_temp_c = excluded.f_temp_c, f_precip_mm = excluded.f_precip_mm,
       f_precip_prob = excluded.f_precip_prob, f_wind_kmh = excluded.f_wind_kmh,
       f_dew_c = excluded.f_dew_c, f_fetched_ts = excluded.f_fetched_ts
     WHERE weather_hourly.f_fetched_ts IS NULL`,
  );
  await writeBatched(
    env,
    hours.map((w) => stmt.bind(w.date, w.hour, w.tempC, w.precipMm, w.precipProb, w.windKmh, w.dewC, now)),
  );
  return hours.length;
}

// Actuals may be rewritten freely — a later ERA5 pass is more authoritative
// than an early forecast-API read of the same past hour.
export async function upsertActuals(env: Env, hours: HourWeather[], now: number): Promise<number> {
  const stmt = env.DB.prepare(
    `INSERT INTO weather_hourly (date, hour, a_temp_c, a_precip_mm, a_wind_kmh, a_dew_c, a_fetched_ts)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(date, hour) DO UPDATE SET
       a_temp_c = excluded.a_temp_c, a_precip_mm = excluded.a_precip_mm,
       a_wind_kmh = excluded.a_wind_kmh, a_dew_c = excluded.a_dew_c,
       a_fetched_ts = excluded.a_fetched_ts`,
  );
  await writeBatched(
    env,
    hours.map((w) => stmt.bind(w.date, w.hour, w.tempC, w.precipMm, w.windKmh, w.dewC, now)),
  );
  return hours.length;
}

// Tomorrow's forecast, plus the tail of tonight — the simulation starts at 10pm
// and runs to noon, so the overnight hours shape the inventory it wakes up with.
export async function fetchForecastFor(env: Env, targetDate: string): Promise<HourWeather[]> {
  const from = addDays(targetDate, -1);
  const all = await fetchHourly(FORECAST_URL, env, from, targetDate, true);
  return all.filter((w) => w.date === targetDate || (w.date === from && w.hour >= 20));
}

// Actuals for past dates: forecast API for recent days, archive for older ones,
// with a forecast-API retry for archive dates that came back all-null (the ERA5
// lag window moves; the forecast API's 92-day past window covers the gap).
export async function fetchActualsFor(env: Env, dates: string[], today: string): Promise<HourWeather[]> {
  const recent = dates.filter((d) => dayDiff(d, today) <= RECENT_DAYS).sort();
  const older = dates.filter((d) => dayDiff(d, today) > RECENT_DAYS).sort();
  const want = new Set(dates);
  const out: HourWeather[] = [];

  if (older.length) {
    const rows = await fetchHourly(ARCHIVE_URL, env, older[0], older[older.length - 1], false);
    const seen = new Map<string, boolean>(); // date -> saw any real value
    for (const w of rows) {
      if (!want.has(w.date)) continue;
      if (w.tempC != null || w.precipMm != null) seen.set(w.date, true);
      else if (!seen.has(w.date)) seen.set(w.date, false);
      out.push(w);
    }
    const missing = [...seen.entries()].filter(([, ok]) => !ok).map(([d]) => d);
    if (missing.length && dayDiff(missing[0], today) <= 92) {
      const retry = await fetchHourly(FORECAST_URL, env, missing[0], missing[missing.length - 1], false);
      const fix = new Set(missing);
      for (const w of retry) if (fix.has(w.date)) out.push(w);
    }
  }
  if (recent.length) {
    const rows = await fetchHourly(FORECAST_URL, env, recent[0], recent[recent.length - 1], false);
    for (const w of rows) if (want.has(w.date)) out.push(w);
  }
  return out;
}

// Dates with facts on the books but no settled actuals yet. Includes today: by
// the 10pm run the morning window is long past, and today's row is what the
// nightly level calibration reads.
export async function datesNeedingActuals(env: Env, today: string): Promise<string[]> {
  const res = await env.DB.prepare(
    `SELECT f.date AS date FROM daily_facts f
     WHERE f.date <= ?
       AND (SELECT COUNT(*) FROM weather_hourly w WHERE w.date = f.date AND w.a_fetched_ts IS NOT NULL) < 24
     ORDER BY f.date ASC`,
  )
    .bind(today)
    .all<{ date: string }>();
  return (res.results ?? []).map((r) => r.date);
}
