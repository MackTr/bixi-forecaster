// HTTP clients for the two sibling services. Both are reached over their public
// /api/v1 contract — this worker never touches another worker's D1, and it does
// not vendor bixi-predictor's model.ts. The control arm being a real HTTP call
// to the real deployed model is the point: a vendored copy could silently drift
// from what bixi-predictor actually published, and then the A/B would be
// comparing this service against a stale fork of its rival rather than the
// rival itself.

import type { Env } from "./worker";

// workers.dev blocks worker→worker fetches within one account, so production
// goes through a service binding (still the public /api/v1 handler, just routed
// directly). Local dev has no bound sibling: wrangler's stub either throws or
// answers 503, and either way the plain fetch of the live URL works fine from
// outside Cloudflare — fall back on both.
async function boundFetch(binding: Fetcher | undefined, url: string): Promise<Response> {
  const init = { headers: { "User-Agent": "bixi-forecaster (personal)" } };
  if (binding) {
    try {
      const res = await binding.fetch(url, init);
      if (res.status !== 503) return res;
    } catch {
      /* fall through to the public URL */
    }
  }
  return fetch(url, init);
}

export interface MonitorObs {
  ts: number;
  bikes: number; // usable (cargo/trailer excluded) — transform applied by the monitor API
}

export async function fetchObservations(env: Env, fromEpoch: number, toEpoch: number): Promise<MonitorObs[]> {
  const url = `${env.MONITOR_API}/stations/${env.STATION_ID}/observations?from=${fromEpoch}&to=${toEpoch}&limit=20000`;
  const res = await boundFetch(env.MONITOR, url);
  if (!res.ok) throw new Error(`monitor api ${res.status}`);
  const body = (await res.json()) as { observations?: MonitorObs[] };
  return body.observations ?? [];
}

// bixi-predictor's published prediction, shaped by its own api.ts. Only the
// fields the mirror and the blend gate need are typed; `basis` carries the two
// the frozen blend rule reads (fallbackLevel, effectiveN).
export interface GaussianPrediction {
  targetDate: string;
  createdAt: string | null;
  probability: number | null;
  predicted: { minutes: number; time: string } | null;
  window: { early: string; late: string } | null;
  basis?: {
    fallbackLevel?: 0 | 1 | 2;
    effectiveN?: number;
    target?: { startBikes?: number | null };
  };
}

// Returns null rather than throwing when the sibling has no prediction for the
// date: on a night bixi-predictor's own cron failed, this service should still
// publish its ml/glm rows and simply record that the control was absent.
export async function fetchGaussianPrediction(env: Env, targetDate: string): Promise<GaussianPrediction | null> {
  const url = `${env.PREDICTOR_API}/stations/${env.STATION_ID}/prediction?date=${targetDate}`;
  const res = await boundFetch(env.PREDICTOR, url);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`predictor api ${res.status}`);
  return (await res.json()) as GaussianPrediction;
}

// "HH:MM" (local) -> minutes since local midnight. bixi-predictor publishes
// window bounds only as strings, so the mirror has to invert its formatter.
export function hhmmToMinutes(s: string | null | undefined): number | null {
  if (!s) return null;
  const m = /^(\d{1,2}):(\d{2})$/.exec(s);
  if (!m) return null;
  return +m[1] * 60 + +m[2];
}
