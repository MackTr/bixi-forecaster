// Upload a trained artifact, prove parity against the exact bytes uploaded, and
// (optionally) activate it.
//
// WHY THIS SCRIPT EXISTS RATHER THAN A CURL COMMAND. The server's
// `/admin/model/parity-passed` endpoint does not and cannot verify parity: the
// golden vectors live on the training machine, and running 453 rows through the
// tree walk is not something a Worker request should do. So that endpoint just
// stamps a timestamp and trusts its caller -- which means the gate is only as
// real as the caller is careful. A human running the three curls by hand can
// stamp parity onto an artifact they never actually tested, and
// `activateArtifact` will happily let it live because all it checks is that the
// stamp exists.
//
// This script IS that careful caller, and it is the only one that should ever
// call parity-passed. The chain it establishes:
//
//   1. read artifact.json as TEXT           -- the bytes that will be uploaded
//   2. parse those bytes, run the gate      -- parity ran on THESE bytes, not on
//                                              whatever is on disk a minute later
//   3. sha256 the text locally
//   4. upload; the server hashes it INDEPENDENTLY and returns its digest
//   5. require the two digests to agree     -- D1 now holds the bytes that passed
//   6. only then stamp parity-passed
//   7. re-read the metadata and require our digest AND the stamp to still be there
//
// Steps 5 and 7 are the load-bearing ones. Without them "parity passed" is a
// claim about a file; with them it is bound to a hash the server computed for
// itself. Step 7 also closes the window in which a concurrent re-upload under
// the same version could have replaced the bytes between 4 and 6 -- storeArtifact
// resets parity_ts on conflict, so a racing upload makes this script fail loudly
// rather than stamp the wrong bytes.
//
// The version is taken FROM THE ARTIFACT, never from a flag. A hand-typed
// --version is precisely how parity gets stamped onto the wrong row.
//
//   node --import ./scripts/ts-ext.mjs scripts/upload-artifact.ts [options]
//
//   --base <url>      service base, default http://127.0.0.1:8789 (wrangler dev)
//   --token <secret>  admin bearer, default $ADMIN_TOKEN
//   --artifact <path> default training/out/artifact.json
//   --golden <path>   default training/out/golden.json
//   --notes <text>    free-text note stored beside the artifact
//   --activate        also make it live (otherwise prints the command to do so)
//   --dry-run         run the gate and hash, touch nothing

import { readFileSync } from "node:fs";

import type { Artifact } from "../src/artifact";
import { DEFAULT_ARTIFACT, DEFAULT_GOLDEN, loadJson, printReport, runParity, type Golden } from "./parity";

interface Options {
  base: string;
  token: string;
  artifact: string;
  golden: string;
  notes: string | null;
  activate: boolean;
  dryRun: boolean;
}

function parseArgs(argv: string[]): Options {
  const o: Options = {
    base: "http://127.0.0.1:8789",
    token: process.env.ADMIN_TOKEN ?? "",
    artifact: DEFAULT_ARTIFACT,
    golden: DEFAULT_GOLDEN,
    notes: null,
    activate: false,
    dryRun: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`${a} expects a value`);
      return v;
    };
    switch (a) {
      case "--base": o.base = next().replace(/\/+$/, ""); break;
      case "--token": o.token = next(); break;
      case "--artifact": o.artifact = next(); break;
      case "--golden": o.golden = next(); break;
      case "--notes": o.notes = next(); break;
      case "--activate": o.activate = true; break;
      case "--dry-run": o.dryRun = true; break;
      // Unknown flags are fatal. Silently ignoring a typo'd --activate would
      // leave the operator believing a model went live when it did not.
      default: throw new Error(`unknown argument ${a}`);
    }
  }
  return o;
}

// Byte-for-byte the same digest the Worker computes in src/artifact.ts. The
// point is not that SHA-256 might differ between the two -- it is that the two
// are computed in different processes over separately-held copies, so agreement
// means the bytes survived the wire and the chunking intact.
async function sha256Hex(text: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

interface ArtifactRow {
  version: string;
  kind: string;
  sha256: string;
  bytes: number;
  chunk_count: number;
  active: number;
  parity_ts: number | null;
}

async function call<T>(o: Options, method: "GET" | "POST", path: string, body?: string): Promise<T> {
  const res = await fetch(`${o.base}/api/v1${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${o.token}`,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body,
  });
  const text = await res.text();
  if (!res.ok) {
    const hint = res.status === 401 ? " (check --token / $ADMIN_TOKEN)" : "";
    throw new Error(`${method} ${path} -> ${res.status}${hint}: ${text.slice(0, 400)}`);
  }
  return JSON.parse(text) as T;
}

const fmtBytes = (n: number) => `${(n / 1024 / 1024).toFixed(2)} MB`;

async function main(): Promise<number> {
  const o = parseArgs(process.argv.slice(2));

  // 1-2. The gate, against the exact bytes destined for D1.
  const text = readFileSync(o.artifact, "utf8");
  const artifact = JSON.parse(text) as Artifact;
  const golden = loadJson<Golden>(o.golden);
  const report = runParity(artifact, golden);
  printReport(report);
  if (!report.passed) {
    console.error(`\nrefusing to upload ${artifact.version} — parity failed`);
    return 1;
  }

  // 3. Our digest of those bytes.
  const localSha = await sha256Hex(text);
  const version = artifact.version;
  console.log(`\nartifact ${version} (${artifact.kind}) — ${fmtBytes(text.length)}, sha256 ${localSha.slice(0, 16)}…`);

  if (o.dryRun) {
    console.log(`dry run — nothing uploaded`);
    return 0;
  }
  if (!o.token) {
    console.error(`no admin token — pass --token or set ADMIN_TOKEN`);
    return 1;
  }

  // Preflight: a 401 or an unreachable host is much friendlier to discover
  // before pushing several megabytes than after.
  const before = await call<{ artifacts: ArtifactRow[]; activeVersion: string | null }>(o, "GET", "/admin/model");
  console.log(`${o.base} — ${before.artifacts.length} artifact(s) stored, active: ${before.activeVersion ?? "none"}`);

  // 4-5. Upload, then require the server's independently-computed digest to
  // match ours. This is what binds the parity verdict to the stored bytes.
  const q = new URLSearchParams({ version, kind: artifact.kind });
  if (o.notes) q.set("notes", o.notes);
  const stored = await call<{ sha256: string; bytes: number; chunks: number }>(o, "POST", `/admin/model/upload?${q}`, text);
  console.log(`uploaded: ${stored.chunks} chunks, ${fmtBytes(stored.bytes)}`);
  if (stored.sha256 !== localSha) {
    console.error(`\nSHA MISMATCH — refusing to stamp parity`);
    console.error(`  local:  ${localSha}`);
    console.error(`  stored: ${stored.sha256}`);
    console.error(`  the bytes in D1 are not the bytes that passed the gate.`);
    return 1;
  }

  // 6. Now, and only now, the stamp.
  await call(o, "POST", `/admin/model/parity-passed?version=${encodeURIComponent(version)}`);
  console.log(`parity recorded for ${version}`);

  // 7. Confirm the stamp landed on a row that still holds our bytes.
  const after = await call<{ artifacts: ArtifactRow[] }>(o, "GET", "/admin/model");
  const row = after.artifacts.find((r) => r.version === version);
  if (!row) throw new Error(`${version} vanished from the artifact list after upload`);
  if (row.sha256 !== localSha) {
    console.error(`\nSHA CHANGED after parity was recorded — something else uploaded ${version} concurrently.`);
    console.error(`  expected ${localSha}\n  found    ${row.sha256}`);
    console.error(`  re-run this script; do NOT activate ${version}.`);
    return 1;
  }
  if (row.parity_ts == null) throw new Error(`parity stamp did not persist for ${version}`);

  if (!o.activate) {
    console.log(`\n${version} is uploaded and verified, but NOT live.`);
    console.log(`  to activate: npm run upload-artifact -- --base ${o.base} --activate`);
    console.log(`  (that re-uploads and re-verifies from scratch, which is the cheap and safe order)`);
    return 0;
  }
  const act = await call<{ activated: string }>(o, "POST", `/admin/model/activate?version=${encodeURIComponent(version)}`);
  console.log(`\nACTIVATED ${act.activated} — it now serves predictions.`);
  return 0;
}

main().then(
  (code) => process.exit(code),
  (e) => {
    console.error(`\nupload-artifact: ${e instanceof Error ? e.message : String(e)}`);
    process.exit(1);
  },
);
