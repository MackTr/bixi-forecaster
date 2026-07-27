// The feature encoder — and the single most dangerous file in the repo.
//
// Everything else either works or throws. An encoder that disagrees with the
// one used at training time produces confident, plausible, wrong numbers,
// forever, silently. That is why scripts/parity.ts feeds ~500 RAW rows (a date
// string and raw millimetres, not a pre-encoded vector) through this file and
// through Python, and why the bucket edges arrive in the artifact instead of
// being written down twice.
//
// The vector is deliberately small and mostly categorical-ish. Measurement on
// 402k station-days said morning rain is strong and monotonic (14.88 -> 11.90
// -> 8.90 mean departures across dry/light/wet) while temperature is weak and
// NON-monotonic — so rain is bucketed coarsely and temperature is handed over
// raw for the trees to carve up however they like.

import type { Artifact } from "./artifact";
import { dowOf } from "./tz";
import { holidayName } from "./holidays";

// Order is the contract. The artifact carries the same list and encode()
// refuses to run against an artifact that disagrees.
export const FEATURE_NAMES = [
  "hour",
  "dow",
  "is_weekend",
  "is_holiday",
  "month",
  "temp_c",
  "precip_bucket",
  "wind_kmh",
  "dew_c",
  "night_precip_bucket",
  "st_morning_share",
  "st_log_level",
  "st_net_balance",
] as const;

export interface RawHour {
  date: string; // local YYYY-MM-DD
  hour: number; // 0..23 local
  tempC: number | null;
  precipMm: number | null; // this hour; label H covers [H-1, H)
  precipProb: number | null; // forecast only — see promotion note below
  windKmh: number | null;
  dewC: number | null;
  nightPrecipMm: number | null; // 20:00-02:00 of the night leading into this date
}

// Coarse buckets, not raw millimetres, and the reason is train/serve skew: at
// 10pm you have a forecast, and a forecast that is 2mm wrong barely moves a
// bucket while it moves a continuous feature a lot. The measured signal was
// already bucket-shaped, so nothing is given up.
function precipBucket(mm: number | null, dryMax: number, wetMin: number): number {
  if (mm == null) return NaN; // uninformative — LightGBM's default_left routes it
  return mm < dryMax ? 0 : mm <= wetMin ? 1 : 2;
}

// precip_prob is NOT a model feature. The archive API has no such field, so
// training could never have seen one; adding it at serve time would be exactly
// the skew this design is trying to avoid. Instead it acts as a deterministic
// promotion on the bucket — the identical rule bixi-predictor applies
// (probBump), so the two services treat a dry-but-likely forecast the same way
// and the comparison isn't confounded by disagreeing about the weather.
function promoteForProb(bucket: number, prob: number | null, threshold: number): number {
  if (bucket === 0 && prob != null && prob >= threshold) return 1;
  return bucket;
}

export function encode(raw: RawHour, a: Artifact): number[] {
  const dow = dowOf(raw.date);
  const month = +raw.date.slice(5, 7);
  const e = a.encoders;
  const bucket = promoteForProb(
    precipBucket(raw.precipMm, e.precipDryMaxMm, e.precipWetMinMm),
    raw.precipProb,
    e.probBumpThreshold,
  );
  return [
    raw.hour,
    dow,
    dow === 0 || dow === 6 ? 1 : 0,
    holidayName(raw.date) ? 1 : 0,
    month,
    raw.tempC ?? NaN,
    bucket,
    raw.windKmh ?? NaN,
    raw.dewC ?? NaN,
    // No probability promotion on the night bucket: precip_prob is a
    // morning-window signal, and bixi-predictor makes the same distinction.
    precipBucket(raw.nightPrecipMm, e.precipDryMaxMm, e.precipWetMinMm),
    a.station.morningShare,
    a.station.logLevel,
    a.station.netBalance,
  ];
}

// Called once when an artifact is loaded. A reordered or renamed feature list is
// the failure mode that would otherwise sail straight past the type system —
// every value is a number, so nothing about a shuffled vector looks wrong.
export function assertFeatureContract(a: Artifact): void {
  if (a.features.length !== FEATURE_NAMES.length) {
    throw new Error(`artifact ${a.version}: ${a.features.length} features, expected ${FEATURE_NAMES.length}`);
  }
  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    if (a.features[i] !== FEATURE_NAMES[i]) {
      throw new Error(`artifact ${a.version}: feature ${i} is "${a.features[i]}", expected "${FEATURE_NAMES[i]}"`);
    }
  }
}
