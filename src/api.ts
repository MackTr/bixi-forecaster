// /api/v1 router. Same conventions as bixi-monitor and bixi-predictor (open
// CORS, JSON errors, station id in the path, Bearer token on admin), and the
// prediction response is deliberately shape-compatible with bixi-predictor's so
// the monitor dashboard can point at either service unchanged.
//
// Deltas: no push routes at all — notifications stay in bixi-predictor for the
// whole shadow period — plus model management, the scoreboard, and replay.

import type { Env } from "./worker";
import { addDays, localToday, minsToHHMM } from "./tz";
import { backfill, runNightly } from "./pipeline";
import { activateArtifact, loadActiveArtifact, markParityPassed, storeArtifact } from "./artifact";
import { scoreboard, type Variant } from "./compare";
import { predictMl } from "./demand";

const CORS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

function json(data: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(data), {
    status: init.status ?? 200,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...CORS,
      ...(init.headers as Record<string, string> | undefined),
    },
  });
}
const fail = (status: number, message: string) => json({ error: message }, { status });
const iso = (epoch: number | null) => (epoch == null ? null : new Date(epoch * 1000).toISOString());

function clampDays(v: string | null, def: number): number {
  const n = parseInt(v ?? "", 10);
  if (!Number.isFinite(n)) return def;
  return Math.min(Math.max(n, 1), 365);
}

const VARIANTS: Variant[] = ["gaussian", "ml", "blend", "glm"];
// `blend` is the published default: it is the arm the frozen gate designates,
// and it degrades to whichever parent is available.
const DEFAULT_VARIANT: Variant = "blend";

function parseVariant(v: string | null): Variant | null {
  if (!v) return DEFAULT_VARIANT;
  return (VARIANTS as string[]).includes(v) ? (v as Variant) : null;
}

// Constant-time-ish Bearer check; refuses everything when the secret is unset.
function authorized(request: Request, env: Env): boolean {
  if (!env.ADMIN_TOKEN) return false;
  const got = request.headers.get("Authorization") ?? "";
  const want = `Bearer ${env.ADMIN_TOKEN}`;
  if (got.length !== want.length) return false;
  let diff = 0;
  for (let i = 0; i < got.length; i++) diff |= got.charCodeAt(i) ^ want.charCodeAt(i);
  return diff === 0;
}

interface PredictionRow {
  target_date: string;
  variant: string;
  created_ts: number;
  predicted_minutes: number | null;
  probability: number | null;
  window_early: number | null;
  window_late: number | null;
  start_bikes: number | null;
  model_version: string | null;
  basis_json: string;
  curve_json: string | null;
  actual_minutes: number | null;
  error_minutes: number | null;
  finalized_ts: number | null;
}

const PRED_COLS =
  "target_date, variant, created_ts, predicted_minutes, probability, window_early, window_late, " +
  "start_bikes, model_version, basis_json, curve_json, actual_minutes, error_minutes, finalized_ts";

function predictionJson(env: Env, r: PredictionRow, opts: { basis?: boolean; curve?: boolean } = {}) {
  const mins = (m: number | null) => (m == null ? null : { minutes: m, time: minsToHHMM(m) });
  return {
    station: env.STATION_ID,
    variant: r.variant,
    targetDate: r.target_date,
    createdAt: iso(r.created_ts),
    modelVersion: r.model_version,
    // null = the model had too little to say anything either way
    willRunOut: r.probability == null ? null : r.predicted_minutes != null,
    probability: r.probability,
    predicted: mins(r.predicted_minutes),
    window:
      r.window_early != null && r.window_late != null
        ? { early: minsToHHMM(r.window_early), late: minsToHHMM(r.window_late) }
        : null,
    startBikes: r.start_bikes,
    actual: mins(r.actual_minutes),
    errorMinutes: r.error_minutes,
    // non-null = the target day is graded; actual:null then means "never ran
    // out", not "not scored yet" — clients cannot tell those apart without this
    finalizedAt: iso(r.finalized_ts),
    ...(opts.basis ? { basis: JSON.parse(r.basis_json) as unknown } : {}),
    ...(opts.curve && r.curve_json ? { curve: JSON.parse(r.curve_json) as unknown } : {}),
  };
}

export async function handleApi(request: Request, env: Env): Promise<Response> {
  if (request.method === "OPTIONS") return new Response(null, { headers: CORS });
  if (request.method !== "GET" && request.method !== "POST") return fail(405, "method not allowed");

  const url = new URL(request.url);
  const parts = url.pathname.replace(/\/+$/, "").split("/").filter(Boolean); // ["api","v1",...]
  if (parts[0] !== "api" || parts[1] !== "v1") return fail(404, "unknown api version");
  const seg = parts.slice(2);
  const GET = request.method === "GET";

  try {
    if (GET && seg.length === 1 && seg[0] === "health") return await health(env);
    if (GET && seg.length === 1 && seg[0] === "compare") {
      return json(await scoreboard(env, clampDays(url.searchParams.get("days"), 60), localToday(Math.floor(Date.now() / 1000))));
    }

    if (seg[0] === "stations" && seg[1]) {
      if (seg[1] !== env.STATION_ID) return fail(404, `unknown station ${seg[1]}`);
      if (GET && seg[2] === "prediction") return await latestPrediction(env, url);
      if (GET && seg[2] === "predictions") return await predictionHistory(env, url);
      return fail(404, `unknown resource ${seg[2] ?? ""}`);
    }

    if (seg[0] === "admin") {
      if (!authorized(request, env)) return fail(401, "unauthorized");
      if (!GET && seg[1] === "backfill") {
        const force = url.searchParams.get("force") === "1";
        return json(await backfill(env, clampDays(url.searchParams.get("days"), 20), { force }));
      }
      if (!GET && seg[1] === "run") return json(await runNightly(env, {}));
      if (!GET && seg[1] === "replay") return await replay(env, url);
      if (seg[1] === "model") return await modelRoutes(env, request, url, seg.slice(2), GET);
      return fail(404, `unknown resource ${seg[1] ?? ""}`);
    }

    return fail(404, "not found");
  } catch (e) {
    return fail(500, e instanceof Error ? e.message : "internal error");
  }
}

// ---------- endpoints ----------

async function latestPrediction(env: Env, url: URL): Promise<Response> {
  const variant = parseVariant(url.searchParams.get("variant"));
  if (!variant) return fail(400, `variant must be one of ${VARIANTS.join(", ")}`);
  const date = url.searchParams.get("date");
  const row = date
    ? await env.DB.prepare(`SELECT ${PRED_COLS} FROM predictions WHERE target_date = ? AND variant = ?`)
        .bind(date, variant)
        .first<PredictionRow>()
    : await env.DB.prepare(`SELECT ${PRED_COLS} FROM predictions WHERE variant = ? ORDER BY target_date DESC LIMIT 1`)
        .bind(variant)
        .first<PredictionRow>();
  if (!row) return fail(404, "no prediction yet");
  const withCurve = url.searchParams.get("curve") === "1";
  return json(predictionJson(env, row, { basis: true, curve: withCurve }));
}

// `all=1` returns every variant for each date side by side — the shape the
// comparison view wants, and the one that makes a divergence between arms
// visible at a glance rather than requiring four requests.
async function predictionHistory(env: Env, url: URL): Promise<Response> {
  const days = clampDays(url.searchParams.get("days"), 14);
  const all = url.searchParams.get("all") === "1";
  const variant = parseVariant(url.searchParams.get("variant"));
  if (!variant) return fail(400, `variant must be one of ${VARIANTS.join(", ")}`);
  const today = localToday(Math.floor(Date.now() / 1000));
  const from = addDays(today, -days);
  const res = all
    ? await env.DB.prepare(`SELECT ${PRED_COLS} FROM predictions WHERE target_date >= ? ORDER BY target_date DESC, variant ASC`)
        .bind(from)
        .all<PredictionRow>()
    : await env.DB.prepare(
        `SELECT ${PRED_COLS} FROM predictions WHERE variant = ? AND target_date >= ? ORDER BY target_date DESC`,
      )
        .bind(variant, from)
        .all<PredictionRow>();
  const rows = res.results ?? [];
  return json({
    station: env.STATION_ID,
    variant: all ? "all" : variant,
    count: rows.length,
    predictions: rows.map((r) => predictionJson(env, r)),
  });
}

// Recompute a past night from the FROZEN forecast columns with the same seeded
// PRNG. Because f_* can never be overwritten and the simulation's dice are
// derived from the date, a replay is structurally incapable of seeing actual
// weather — the proof is the schema, not a promise in a comment.
async function replay(env: Env, url: URL): Promise<Response> {
  const date = url.searchParams.get("date");
  if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) return fail(400, "expected ?date=YYYY-MM-DD");
  const before = await env.DB.prepare(
    `SELECT predicted_minutes, probability, window_early, window_late FROM predictions WHERE target_date = ? AND variant = 'ml'`,
  )
    .bind(date)
    .first<{ predicted_minutes: number | null; probability: number | null; window_early: number | null; window_late: number | null }>();
  const now = Math.floor(Date.now() / 1000);
  const after = await predictMl(env, date, now);
  return json({
    targetDate: date,
    stored: before ?? null,
    replayed: after,
    // A mismatch means something non-deterministic leaked into the path — a
    // changed artifact, a rewritten forecast, or a clock-seeded draw.
    identical:
      !!before && !!after && before.predicted_minutes === after.predictedMinutes && before.probability === after.probability,
  });
}

async function modelRoutes(env: Env, request: Request, url: URL, seg: string[], GET: boolean): Promise<Response> {
  if (GET && seg.length === 0) {
    const res = await env.DB.prepare(
      `SELECT version, kind, sha256, bytes, chunk_count, active, parity_ts, notes, created_ts
       FROM model_artifacts ORDER BY created_ts DESC LIMIT 20`,
    ).all<Record<string, unknown>>();
    const active = await loadActiveArtifact(env).catch((e) => ({ version: `ERROR: ${e.message}` }) as never);
    return json({ artifacts: res.results ?? [], activeVersion: active?.version ?? null });
  }
  if (!GET && seg[0] === "upload") {
    const version = url.searchParams.get("version");
    if (!version) return fail(400, "expected ?version=");
    const kind = (url.searchParams.get("kind") ?? "gbdt") as "gbdt" | "glm";
    const body = await request.text();
    if (!body) return fail(400, "empty body");
    try {
      JSON.parse(body);
    } catch {
      return fail(400, "body is not valid JSON");
    }
    return json(await storeArtifact(env, version, kind, body, url.searchParams.get("notes"), Math.floor(Date.now() / 1000)));
  }
  // Parity is asserted by scripts/parity.ts running under Node against the same
  // bytes; this route only records that verdict. It is separate from activation
  // on purpose — activation refuses without it.
  if (!GET && seg[0] === "parity-passed") {
    const version = url.searchParams.get("version");
    if (!version) return fail(400, "expected ?version=");
    await markParityPassed(env, version, Math.floor(Date.now() / 1000));
    return json({ ok: true, version });
  }
  if (!GET && seg[0] === "activate") {
    const version = url.searchParams.get("version");
    if (!version) return fail(400, "expected ?version=");
    return json(await activateArtifact(env, version));
  }
  return fail(404, `unknown resource ${seg[0] ?? ""}`);
}

async function health(env: Env): Promise<Response> {
  const now = Math.floor(Date.now() / 1000);
  const today = localToday(now);
  const latest = await env.DB.prepare(
    `SELECT target_date, MAX(created_ts) AS created_ts, COUNT(*) AS variants
     FROM predictions GROUP BY target_date ORDER BY target_date DESC LIMIT 1`,
  ).first<{ target_date: string; created_ts: number; variants: number }>();
  const facts = await env.DB.prepare(`SELECT COUNT(*) AS c, MAX(date) AS latest FROM daily_facts`).first<{
    c: number;
    latest: string | null;
  }>();
  const hourly = await env.DB.prepare(`SELECT COUNT(*) AS c FROM hourly_facts`).first<{ c: number }>();
  const weather = await env.DB.prepare(
    `SELECT COUNT(*) AS c, SUM(CASE WHEN f_fetched_ts IS NOT NULL THEN 1 ELSE 0 END) AS frozen FROM weather_hourly`,
  ).first<{ c: number; frozen: number }>();
  const model = await env.DB.prepare(
    `SELECT version FROM model_artifacts WHERE active = 1 AND parity_ts IS NOT NULL LIMIT 1`,
  ).first<{ version: string }>();
  return json({
    // healthy = last night's run produced predictions for today or later
    ok: latest != null && latest.target_date >= today,
    latestTargetDate: latest?.target_date ?? null,
    latestCreatedAt: iso(latest?.created_ts ?? null),
    variantsLastNight: latest?.variants ?? 0,
    factDays: facts?.c ?? 0,
    latestFactDate: facts?.latest ?? null,
    hourlyRows: hourly?.c ?? 0,
    weatherHours: weather?.c ?? 0,
    weatherHoursFrozen: weather?.frozen ?? 0,
    activeModel: model?.version ?? null,
    // Explicit, because "no model yet" is the expected state for the first
    // weeks and should not read as a fault.
    mode: model?.version ? "shadow" : "collecting",
    serverTime: iso(now),
  });
}
