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
//
// Run directly (`npm run parity`) this checks the artifact on disk and exits
// non-zero on failure. It is ALSO imported by upload-artifact.ts, which runs the
// gate against the exact bytes it is about to upload. That reuse is deliberate:
// a second parity implementation living in the uploader could drift from this
// one, and the whole point of a gate is that there is only one of it.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Artifact, Booster, FlatTree } from "../src/artifact";
import { assertFeatureContract, encode, FEATURE_NAMES, type RawHour } from "../src/features";
import { predict } from "../src/gbdt";

const HERE = dirname(fileURLToPath(import.meta.url));
export const OUT = join(HERE, "..", "training", "out");
export const DEFAULT_ARTIFACT = join(OUT, "artifact.json");
export const DEFAULT_GOLDEN = join(OUT, "golden.json");
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

export interface Golden {
  version: string;
  features: string[];
  encoders: Artifact["encoders"];
  station: Artifact["station"];
  rows: GoldenRow[];
}

export interface ParityReport {
  passed: boolean;
  version: string;
  rows: number;
  nanRows: number;
  promotedRows: number;
  worstEncode: number;
  worstDep: number;
  worstArr: number;
  encodeMismatches: number;
  predMismatches: number;
  problems: string[];
  // What fraction of the model the golden rows actually touched. See
  // leafCoverage() for why this is reported rather than asserted.
  coverage: { departures: Coverage; arrivals: Coverage };
}

interface Coverage {
  hit: number;
  total: number;
}

// Which leaf each tree lands on, for coverage accounting only.
//
// This duplicates the descent in src/gbdt.ts, which is normally exactly the
// mistake this gate exists to prevent -- but it returns a NODE INDEX rather than
// a value and never contributes to the pass/fail verdict. If it ever diverged
// from gbdt.ts the coverage percentage would be wrong; the gate would not.
function leafIndex(t: FlatTree, x: number[]): number {
  let i = 0;
  for (let guard = 0; guard < t.feature.length + 1; guard++) {
    const f = t.feature[i];
    if (f < 0) return i;
    const v = x[f];
    i = Number.isNaN(v) ? (t.defaultLeft[i] ? t.left[i] : t.right[i]) : v <= t.threshold[i] ? t.left[i] : t.right[i];
  }
  throw new Error("parity: tree walk exceeded node count — malformed artifact");
}

// The gate proves Python and TypeScript agree ON THE PATHS THE GOLDEN ROWS TAKE.
// It says nothing about leaves no row reaches, and at 453 rows that is most of
// them. That is the right scope for what parity is FOR -- an encoder bug is
// systematic and shifts many rows at once, which is why rounding every leaf to
// 1e-9 once failed 136 of these 453 rows -- but it means parity alone is not an
// integrity check on the artifact as a whole. A single wrong leaf value in the
// unvisited majority passes.
//
// What covers that gap is the SHA-256 chain, not this: upload-artifact.ts
// requires the digest D1 computed to equal the digest of the bytes fed to this
// gate, and loadActiveArtifact re-verifies on every cold load. Between them,
// transport and storage corruption cannot survive. What neither covers is an
// error introduced upstream in export.py itself -- so the number is printed on
// every run rather than left to be assumed.
function leafCoverage(b: Booster, encoded: number[][]): Coverage {
  let total = 0;
  let hit = 0;
  // Counted per tree and summed, rather than with one composite key across all
  // trees, so nothing depends on an assumed ceiling for node counts.
  for (const t of b.trees) {
    for (const f of t.feature) if (f < 0) total++;
    const seen = new Set<number>();
    for (const x of encoded) seen.add(leafIndex(t, x));
    hit += seen.size;
  }
  return { hit, total };
}

export function loadJson<T>(path: string): T {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as T;
  } catch (e) {
    console.error(`parity: cannot read ${path}\n  run: cd training && .venv/bin/python -m bixi_train.export`);
    throw e;
  }
}

// Takes a PARSED artifact rather than a path, so a caller that is about to
// upload some specific bytes can prove the gate ran on those bytes and not on
// whatever happens to be sitting on disk at the moment.
export function runParity(artifact: Artifact, golden: Golden): ParityReport {
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
  const encoded: number[][] = [];

  for (let i = 0; i < golden.rows.length; i++) {
    const g = golden.rows[i];
    const raw: RawHour = g.raw;
    const x = encode(raw, artifact);
    encoded.push(x);

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

  return {
    passed: problems.length === 0,
    version: artifact.version,
    rows: golden.rows.length,
    nanRows: golden.rows.filter((r) => r.encoded.some((v) => v === null)).length,
    promotedRows: golden.rows.filter((r) => r.raw.precipProb != null).length,
    worstEncode,
    worstDep,
    worstArr,
    encodeMismatches,
    predMismatches,
    problems,
    coverage: {
      departures: leafCoverage(artifact.departures, encoded),
      arrivals: leafCoverage(artifact.arrivals, encoded),
    },
  };
}

// Shared by both entry points so the uploader's transcript and `npm run parity`
// report the same numbers in the same words.
export function printReport(r: ParityReport): void {
  console.log(`parity: ${r.rows} golden rows against ${r.version}`);
  console.log(`  coverage: ${r.nanRows} rows with missing measurements, ${r.promotedRows} exercising the precip-probability bump`);
  console.log(
    `  max diff: encode ${r.worstEncode.toExponential(3)}, departures ${r.worstDep.toExponential(3)}, arrivals ${r.worstArr.toExponential(3)}`,
  );
  const pct = (c: Coverage) => `${c.hit}/${c.total} (${((100 * c.hit) / c.total).toFixed(0)}%)`;
  console.log(`  leaves reached: departures ${pct(r.coverage.departures)}, arrivals ${pct(r.coverage.arrivals)}`);
  console.log(`  (unreached leaves are unverified by this gate — the sha256 chain, not parity, is what protects them)`);
  if (r.passed) return;
  console.error(`\nparity FAILED (${r.encodeMismatches} encode, ${r.predMismatches} prediction mismatches):`);
  for (const p of r.problems.slice(0, 12)) console.error(`  ${p}`);
  if (r.problems.length > 12) console.error(`  ... and ${r.problems.length - 12} more`);
}

function main(): number {
  const report = runParity(loadJson<Artifact>(DEFAULT_ARTIFACT), loadJson<Golden>(DEFAULT_GOLDEN));
  printReport(report);
  if (!report.passed) return 1;
  console.log(`\nparity PASSED — ${report.version} may be activated`);
  return 0;
}

// Only when run as the script, not when imported by the uploader.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  process.exit(main());
}
