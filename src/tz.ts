// Timezone + local-date helpers shared by sync and the pipeline. localParts /
// wallToEpoch / bucketParts are lifted from bixi-monitor's analytics.ts along
// with its hard-learned rule: constructing an Intl.DateTimeFormat per call blew
// the Workers CPU budget there (1102 errors on /stats), so formatters are cached
// per timezone and wall-clock facts are memoized per quarter-hour bucket.

export const TZ = "America/Toronto";

const WD = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

export interface LocalParts {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  weekday: number; // 0=Sun..6=Sat
  dateStr: string; // YYYY-MM-DD (local)
}

const FMT_CACHE = new Map<string, Intl.DateTimeFormat>();
function formatterFor(tz: string): Intl.DateTimeFormat {
  let f = FMT_CACHE.get(tz);
  if (!f) {
    f = new Intl.DateTimeFormat("en-CA", {
      timeZone: tz,
      hour12: false,
      weekday: "short",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    FMT_CACHE.set(tz, f);
  }
  return f;
}

export function localParts(epoch: number, tz: string = TZ): LocalParts {
  const fmt = formatterFor(tz);
  const p: Record<string, string> = {};
  for (const part of fmt.formatToParts(new Date(epoch * 1000))) p[part.type] = part.value;
  const hour = parseInt(p.hour, 10) % 24; // some platforms emit "24" at midnight
  return {
    year: +p.year,
    month: +p.month,
    day: +p.day,
    hour,
    minute: +p.minute,
    second: +p.second,
    weekday: WD.indexOf(p.weekday),
    dateStr: `${p.year}-${p.month}-${p.day}`,
  };
}

// Inverse of localParts: epoch for a given local wall-clock time. Converges in a
// couple of iterations and is DST-correct away from the transition instant.
export function wallToEpoch(y: number, m: number, d: number, hh: number, mm: number, tz: string = TZ): number {
  let guess = Math.floor(Date.UTC(y, m - 1, d, hh, mm, 0) / 1000);
  for (let i = 0; i < 3; i++) {
    const p = localParts(guess, tz);
    const shown = Math.floor(Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second) / 1000);
    const want = Math.floor(Date.UTC(y, m - 1, d, hh, mm, 0) / 1000);
    guess += want - shown;
  }
  return guess;
}

// Local weekday/hour/date are constant within a quarter-hour (every real UTC
// offset is a multiple of 15 min), so the Intl lookup memoizes per bucket.
const BUCKET_PARTS = new Map<string, LocalParts>();
export function bucketParts(epoch: number, tz: string = TZ): LocalParts {
  const k = Math.floor(epoch / 900);
  const key = `${tz}:${k}`;
  let p = BUCKET_PARTS.get(key);
  if (!p) {
    if (BUCKET_PARTS.size > 50_000) BUCKET_PARTS.clear();
    p = localParts(k * 900, tz);
    BUCKET_PARTS.set(key, p);
  }
  return p;
}

// ---------- pure date-string helpers (no Intl at all) ----------

const pad = (n: number) => String(n).padStart(2, "0");

// Local calendar date "now". The nightly cron fires at 01:00/02:00 UTC when the
// UTC date is already tomorrow — every date in the pipeline must come from here.
export function localToday(nowEpoch: number): string {
  return localParts(nowEpoch, TZ).dateStr;
}

export function addDays(dateStr: string, n: number): string {
  const [y, m, d] = dateStr.split("-").map(Number);
  const t = new Date(Date.UTC(y, m - 1, d + n));
  return `${t.getUTCFullYear()}-${pad(t.getUTCMonth() + 1)}-${pad(t.getUTCDate())}`;
}

export function dowOf(dateStr: string): number {
  const [y, m, d] = dateStr.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).getUTCDay();
}

// Calendar days from a to b (positive when b is later).
export function dayDiff(a: string, b: string): number {
  const [ay, am, ad] = a.split("-").map(Number);
  const [by, bm, bd] = b.split("-").map(Number);
  return Math.round((Date.UTC(by, bm - 1, bd) - Date.UTC(ay, am - 1, ad)) / 86_400_000);
}

// Epoch of local midnight starting the given local date.
export function midnightEpoch(dateStr: string): number {
  const [y, m, d] = dateStr.split("-").map(Number);
  return wallToEpoch(y, m, d, 0, 0, TZ);
}

export function minsToHHMM(mins: number): string {
  return `${pad(Math.floor(mins / 60) % 24)}:${pad(mins % 60)}`;
}
