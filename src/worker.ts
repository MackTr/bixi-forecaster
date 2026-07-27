import { handleApi } from "./api";
import { runNightly } from "./pipeline";
import { TZ, localParts } from "./tz";

export interface Env {
  DB: D1Database;
  MONITOR?: Fetcher; // service binding to bixi-monitor (prod; absent-ish in local dev)
  PREDICTOR?: Fetcher; // service binding to bixi-predictor — the control arm
  MONITOR_API: string; // var — bixi-monitor's /api/v1 base URL (dev fallback + URL builder)
  PREDICTOR_API: string; // var — bixi-predictor's /api/v1 base URL
  STATION_ID: string; // var
  STATION_LAT: string; // var
  STATION_LON: string; // var
  STATION_CAPACITY: string; // var — docks at station 345; caps simulated inventory
  ADMIN_TOKEN: string; // secret
}

export default {
  // API only — no dashboard here; bixi-monitor's dashboard is the UI, and it
  // consumes this service through the same response shape bixi-predictor
  // already publishes.
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api" || url.pathname.startsWith("/api/")) {
      return handleApi(request, env);
    }
    return new Response(JSON.stringify({ error: "not found", api: "/api/v1" }), {
      status: 404,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  },

  // Both UTC crons (02:05 and 03:05) fire year-round; only the one landing at
  // 10pm Montreal time does work, which keeps the schedule DST-proof. Errors are
  // logged, never thrown, so one bad night can't wedge the schedule.
  //
  // NOTE: no push. Notifications stay in bixi-predictor for the whole shadow
  // period — two workers notifying the same subscriptions would confound the
  // very comparison this service exists to run.
  async scheduled(_event: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    const hour = localParts(Math.floor(Date.now() / 1000), TZ).hour;
    if (hour !== 22) {
      console.log(`cron skipped (local hour ${hour} != 22)`);
      return;
    }
    ctx.waitUntil(
      runNightly(env, {}).then(
        (result) => console.log("nightly:", JSON.stringify(result)),
        (err) => console.error("nightly failed:", err instanceof Error ? err.message : err),
      ),
    );
  },
} satisfies ExportedHandler<Env>;
