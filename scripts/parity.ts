// The gate. No artifact may be activated until this passes.
//
// Training happens in Python; inference happens in TypeScript. Those are two
// independent implementations of the same arithmetic, and the failure mode when
// they disagree is not a crash -- it is a service that keeps answering, keeps
// looking healthy, and is quietly wrong for as long as nobody checks. Types
// cannot catch it: every value on both sides is a number.
//
// So export.py emits ~500 RAW rows -- a date string and millimetres of rain, not
// a pre-encoded vector -- together with what Python computed from them. This
// runs the same raw rows through src/features.ts and src/gbdt.ts and requires
// agreement to 1e-9.
//
// Feeding raw rows rather than encoded ones is the entire point. The encoder is
// where the bugs live: bucket edges, the 0=Sunday convention, holiday dates, the
// night-precipitation window, NaN handling. Handing TypeScript a vector Python
// had already encoded would test the tree walk -- the one part that is nearly
// impossible to get wrong -- and skip everything that is easy to get wrong.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Artifact } from "../src/artifact";
import { assertFeatureContract, encode, FEATURE_NAMES, type RawHour } from "../src/features";
import { predict } from "../src/gbdt";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "..", "training", "out");
const TOL = 1e-9;

interface GoldenRow {
  raw: {
    date: string;
    hour: number;
    tempC: number | null;
    precipMm: number | null;
    precipProb: number | null;
    windKmh: number | null;
    dewC: number | null;
    nightPrecipMm: number | null;
  };
  encoded: (number | null)[]; // null encodes NaN -- JSON has no NaN literal
  departures: number;
  arrivals: number;
}

interface Golden {
  version: string;
  features: string[];
  encoders: Artifact["encoders"];
  station: Artifact["station"];
  rows: GoldenRow[];
}

function load<T>(name: string): T {
  const p = join(OUT, name);
  try {
    return JSON.parse(readFileSync(p, "utf8")) as T;
  } catch (e) {
    console.error(`parity: cannot read ${p}\n  run: cd training && .venv/bin/python -m bixi_train.export`);
    throw e;
  }
}

function main(): number {
  const artifact = load<Artifact>("artifact.json");
  const golden = load<Golden>("golden.json");

  const problems: string[] = [];
  const fail = (m: string) => problems.push(m);

  // The goldens must describe the artifact under test. A stale pair would let a
  // model activate on evidence gathered about a different model.
  if (golden.version !== artifact.version) {
    fail(`version mismatch: artifact ${artifact.version} vs golden ${golden.version}`);
  }
  assertFeatureContract(artifact);
  if (golden.features.join(",") !== FEATURE_NAMES.join(",")) {
    fail(`golden feature order ${golden.features.join(",")} != ${FEATURE_NAMES.join(",")}`);
  }
  for (const k of ["precipDryMaxMm", "precipWetMinMm", "probBumpThreshold"] as const) {
    if (artifact.encoders[k] !== golden.encoders[k]) {
      fail(`encoder ${k}: artifact ${artifact.encoders[k]} vs golden ${golden.encoders[k]}`);
    }
  }

  let worstEncode = 0;
  let worstDep = 0;
  let worstArr = 0;
  let encodeMismatches = 0;
  let predMismatches = 0;

  for (let i = 0; i < golden.rows.length; i++) {
    const g = golden.rows[i];
    const raw: RawHour = g.raw;
    const x = encode(raw, artifact);

    for (let j = 0; j < FEATURE_NAMES.length; j++) {
      const want = g.encoded[j];
      const got = x[j];
      // NaN is a legitimate encoded value (missing measurement), and it must
      // agree as NaN on both sides rather than being silently coerced.
      if (want === null) {
        if (!Number.isNaN(got)) {
          if (encodeMismatches++ < 5) fail(`row ${i} feature ${FEATURE_NAMES[j]}: expected NaN, got ${got}`);
        }
        continue;
      }
      if (Number.isNaN(got)) {
        if (encodeMismatches++ < 5) fail(`row ${i} feature ${FEATURE_NAMES[j]}: expected ${want}, got NaN`);
        continue;
      }
      const d = Math.abs(got - want);
      if (d > worstEncode) worstEncode = d;
      if (d > TOL && encodeMismatches++ < 5) {
        fail(`row ${i} feature ${FEATURE_NAMES[j]}: expected ${want}, got ${got} (diff ${d.toExponential(3)})`);
      }
    }

    for (const [which, want, track] of [
      ["departures", g.departures, (d: number) => (worstDep = Math.max(worstDep, d))],
      ["arrivals", g.arrivals, (d: number) => (worstArr = Math.max(worstArr, d))],
    ] as const) {
      const got = predict(artifact[which], x);
      // Relative tolerance: these are exponentials, and an absolute epsilon
      // would be vacuously easy on small rates and impossibly strict on large.
      const d = Math.abs(got - want) / Math.max(1, Math.abs(want));
      track(d);
      if (d > TOL && predMismatches++ < 5) {
        fail(`row ${i} ${which}: expected ${want}, got ${got} (rel diff ${d.toExponential(3)})`);
      }
    }
  }

  const nan = golden.rows.filter((r) => r.encoded.some((v) => v === null)).length;
  const promoted = golden.rows.filter((r) => r.raw.precipProb != null).length;
  console.log(`parity: ${golden.rows.length} golden rows against ${artifact.version}`);
  console.log(`  coverage: ${nan} rows with missing measurements, ${promoted} exercising the precip-probability bump`);
  console.log(`  max diff: encode ${worstEncode.toExponential(3)}, departures ${worstDep.toExponential(3)}, arrivals ${worstArr.toExponential(3)}`);

  if (problems.length) {
    console.error(`\nparity FAILED (${encodeMismatches} encode, ${predMismatches} prediction mismatches):`);
    for (const p of problems.slice(0, 12)) console.error(`  ${p}`);
    if (problems.length > 12) console.error(`  ... and ${problems.length - 12} more`);
    return 1;
  }
  console.log(`\nparity PASSED — ${artifact.version} may be activated`);
  return 0;
}

process.exit(main());
