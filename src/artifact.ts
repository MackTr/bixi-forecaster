// Model artifacts: versioned, chunked, SHA-256 verified, loaded from D1.
//
// Python cannot run in a Worker, so training is offline and inference is
// TypeScript. The bridge is this file: LightGBM trees arrive as flat numeric
// arrays that gbdt.ts walks directly. ONNX Runtime Web was the obvious
// alternative and does not work here — Workers cannot instantiate WASM from
// bytes fetched at runtime, and ORT's bundle would dwarf the size budget for a
// model a tree walk evaluates EXACTLY rather than approximately.
//
// The artifact lives in D1 rather than the bundle so that retraining and
// rollback are database writes, not redeploys, and so the exact bytes behind any
// past prediction stay recoverable.

import type { Env } from "./worker";

// One flattened decision tree. Node i is a leaf when feature[i] < 0, in which
// case value[i] is its output; otherwise the split is `x[feature[i]] <=
// threshold[i]`, NaN routed by defaultLeft. Parallel arrays rather than objects
// because a few thousand small objects is the kind of allocation churn that
// shows up as CPU time on a Worker.
export interface FlatTree {
  feature: number[];
  threshold: number[];
  left: number[];
  right: number[];
  value: number[];
  defaultLeft: number[];
}

export interface Booster {
  // "poisson" -> exp(raw); "regression" -> raw. LightGBM folds its
  // boost-from-average init score into the first tree, so there is no separate
  // bias term to add — a fact the parity gate verifies rather than assumes.
  objective: "poisson" | "regression";
  trees: FlatTree[];
}

export interface StationProfile {
  morningShare: number; // share of the station's daily departures falling in 6-11am
  logLevel: number; // log mean daily 6-11am departures over the training seasons
  netBalance: number; // (arrivals - departures) / (arrivals + departures)
}

export interface Artifact {
  version: string;
  kind: "gbdt" | "glm";
  createdAt: string;
  features: string[]; // ordered feature names — features.ts asserts against this
  // Bucket edges SHIP HERE rather than being hardcoded in TypeScript. If they
  // lived in both places, a retrain that moved an edge would produce a Worker
  // that silently encodes its inputs differently than the model was trained on,
  // and nothing would fail loudly.
  encoders: {
    precipDryMaxMm: number;
    precipWetMinMm: number;
    probBumpThreshold: number; // forecast rain probability that promotes dry -> light
  };
  station: StationProfile;
  departures: Booster;
  arrivals: Booster;
  // GLM arm: one coefficient vector per target over the same encoded features.
  glm?: { departures: { intercept: number; coef: number[] }; arrivals: { intercept: number; coef: number[] } };
}

async function sha256Hex(text: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Chunk size well under D1's per-value ceiling. Chunks are plain text slices of
// the JSON, so reassembly is a join and needs no framing format of its own.
const CHUNK = 256 * 1024;

export async function storeArtifact(
  env: Env,
  version: string,
  kind: "gbdt" | "glm",
  json: string,
  notes: string | null,
  now: number,
): Promise<{ version: string; sha256: string; bytes: number; chunks: number }> {
  const sha = await sha256Hex(json);
  const chunks: string[] = [];
  for (let i = 0; i < json.length; i += CHUNK) chunks.push(json.slice(i, i + CHUNK));

  const stmts: D1PreparedStatement[] = [
    env.DB.prepare(`DELETE FROM model_artifact_chunks WHERE version = ?`).bind(version),
    env.DB.prepare(
      `INSERT INTO model_artifacts (version, kind, sha256, bytes, chunk_count, active, parity_ts, notes, created_ts)
       VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?)
       ON CONFLICT(version) DO UPDATE SET
         kind = excluded.kind, sha256 = excluded.sha256, bytes = excluded.bytes,
         chunk_count = excluded.chunk_count, active = 0, parity_ts = NULL,
         notes = excluded.notes, created_ts = excluded.created_ts`,
    ).bind(version, kind, sha, json.length, chunks.length, notes, now),
  ];
  const chunkStmt = env.DB.prepare(`INSERT INTO model_artifact_chunks (version, idx, body) VALUES (?, ?, ?)`);
  chunks.forEach((body, i) => stmts.push(chunkStmt.bind(version, i, body)));
  // Re-uploading always lands as inactive with parity_ts cleared: new bytes
  // have not passed the gate yet, whatever the previous bytes under this
  // version had proved.
  for (let i = 0; i < stmts.length; i += 20) await env.DB.batch(stmts.slice(i, i + 20));
  return { version, sha256: sha, bytes: json.length, chunks: chunks.length };
}

// Module-level cache: a Worker isolate serves many requests, and re-parsing a
// multi-megabyte artifact per prediction would dominate the CPU budget.
let cached: { version: string; artifact: Artifact } | null = null;

export async function loadActiveArtifact(env: Env, kind: "gbdt" | "glm" = "gbdt"): Promise<Artifact | null> {
  const meta = await env.DB.prepare(
    `SELECT version, sha256, chunk_count FROM model_artifacts
     WHERE kind = ? AND active = 1 AND parity_ts IS NOT NULL
     ORDER BY created_ts DESC LIMIT 1`,
  )
    .bind(kind)
    .first<{ version: string; sha256: string; chunk_count: number }>();
  if (!meta) return null;
  if (cached?.version === meta.version) return cached.artifact;

  const res = await env.DB.prepare(`SELECT body FROM model_artifact_chunks WHERE version = ? ORDER BY idx ASC`)
    .bind(meta.version)
    .all<{ body: string }>();
  const parts = res.results ?? [];
  if (parts.length !== meta.chunk_count) {
    throw new Error(`artifact ${meta.version}: ${parts.length}/${meta.chunk_count} chunks`);
  }
  const json = parts.map((p) => p.body).join("");
  // Verified on every cold load, not just at upload: a partial write or a
  // truncated chunk would otherwise surface as subtly wrong predictions rather
  // than as an error.
  const sha = await sha256Hex(json);
  if (sha !== meta.sha256) throw new Error(`artifact ${meta.version}: sha256 mismatch`);

  const artifact = JSON.parse(json) as Artifact;
  cached = { version: meta.version, artifact };
  return artifact;
}

// Activation is a separate, explicit act from upload, and it refuses to proceed
// unless the golden-vector parity run has signed off on these exact bytes.
export async function activateArtifact(env: Env, version: string): Promise<{ activated: string }> {
  const row = await env.DB.prepare(`SELECT kind, parity_ts FROM model_artifacts WHERE version = ?`)
    .bind(version)
    .first<{ kind: string; parity_ts: number | null }>();
  if (!row) throw new Error(`unknown artifact ${version}`);
  if (row.parity_ts == null) throw new Error(`artifact ${version} has not passed parity — refusing to activate`);
  await env.DB.batch([
    env.DB.prepare(`UPDATE model_artifacts SET active = 0 WHERE kind = ?`).bind(row.kind),
    env.DB.prepare(`UPDATE model_artifacts SET active = 1 WHERE version = ?`).bind(version),
  ]);
  cached = null;
  return { activated: version };
}

export async function markParityPassed(env: Env, version: string, now: number): Promise<void> {
  await env.DB.prepare(`UPDATE model_artifacts SET parity_ts = ? WHERE version = ?`).bind(now, version).run();
}
