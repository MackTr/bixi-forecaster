// Quebec statutory holidays (Act respecting labour standards), computed rather
// than maintained as a list. Pure calendar math over UTC dates — callers match
// against local YYYY-MM-DD strings, so no timezone handling belongs here.
// Consciously skipped: observed-Monday shifts for weekend holidays (weekends are
// already excluded from weekday stats), Dec 24/26/31, the construction holiday.

const DAY_MS = 86_400_000;
const utc = (y: number, m: number, d: number) => new Date(Date.UTC(y, m - 1, d));
const plusDays = (d: Date, n: number) => new Date(d.getTime() + n * DAY_MS);
const ymd = (d: Date) =>
  `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-${String(d.getUTCDate()).padStart(2, "0")}`;

// Anonymous Gregorian computus (Meeus).
export function easterSunday(year: number): Date {
  const a = year % 19;
  const b = Math.floor(year / 100);
  const c = year % 100;
  const d = Math.floor(b / 4);
  const e = b % 4;
  const f = Math.floor((b + 8) / 25);
  const g = Math.floor((b - f + 1) / 3);
  const h = (19 * a + b - d - g + 15) % 30;
  const i = Math.floor(c / 4);
  const k = c % 4;
  const l = (32 + 2 * e + 2 * i - h - k) % 7;
  const m = Math.floor((a + 11 * h + 22 * l) / 451);
  const month = Math.floor((h + l - 7 * m + 114) / 31);
  const day = ((h + l - 7 * m + 114) % 31) + 1;
  return utc(year, month, day);
}

function nthMonday(year: number, month: number, n: number): Date {
  const first = utc(year, month, 1);
  return plusDays(first, ((8 - first.getUTCDay()) % 7) + (n - 1) * 7);
}

export function quebecHolidays(year: number): Map<string, string> {
  const out = new Map<string, string>();
  const add = (d: Date, name: string) => out.set(ymd(d), name);
  add(utc(year, 1, 1), "New Year's Day");
  // The statute grants Good Friday OR Easter Monday at the employer's choice,
  // so commuter traffic is light on both — count both.
  const easter = easterSunday(year);
  add(plusDays(easter, -2), "Good Friday");
  add(plusDays(easter, 1), "Easter Monday");
  // Monday strictly preceding May 25 (lands May 18–24).
  const may25 = utc(year, 5, 25);
  add(plusDays(may25, -(((may25.getUTCDay() + 6) % 7) || 7)), "National Patriots' Day");
  add(utc(year, 6, 24), "Fête nationale");
  const jul1 = utc(year, 7, 1);
  add(jul1.getUTCDay() === 0 ? utc(year, 7, 2) : jul1, "Canada Day"); // QC observes Jul 2 when the 1st is a Sunday
  add(nthMonday(year, 9, 1), "Labour Day");
  add(nthMonday(year, 10, 2), "Thanksgiving");
  add(utc(year, 12, 25), "Christmas Day");
  return out;
}

const byYear = new Map<number, Map<string, string>>();

// dateStr = local YYYY-MM-DD, as produced by analytics' localParts.
export function holidayName(dateStr: string): string | null {
  const year = +dateStr.slice(0, 4);
  let map = byYear.get(year);
  if (!map) {
    map = quebecHolidays(year);
    byYear.set(year, map);
  }
  return map.get(dateStr) ?? null;
}
