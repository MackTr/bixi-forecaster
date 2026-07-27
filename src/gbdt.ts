// A dependency-free gradient-boosted-tree evaluator: ~40 lines of tree walk
// standing in for a Python runtime that cannot exist here.
//
// This is exact, not approximate. A decision tree's prediction is a sequence of
// comparisons and a leaf lookup; there is no linear algebra to be off by an
// epsilon on. That is why the ONNX route was rejected rather than merely
// deferred — the runtime it would drag in buys nothing a comparison operator
// doesn't already do.
//
// The correctness argument is entirely in scripts/parity.ts: ~500 RAW feature
// rows go through Python and through this file, and the artifact does not
// activate unless every output matches to 1e-9.

import type { Booster, FlatTree } from "./artifact";

function walk(t: FlatTree, x: number[]): number {
  let i = 0;
  // A malformed tree could otherwise spin forever; depth is bounded by the
  // node count, so exceeding it means the arrays are not a tree.
  for (let guard = 0; guard < t.feature.length + 1; guard++) {
    const f = t.feature[i];
    if (f < 0) return t.value[i];
    const v = x[f];
    // NaN fails both comparisons, so it must be routed explicitly — and it must
    // be routed the way LightGBM routed it during training.
    i = Number.isNaN(v) ? (t.defaultLeft[i] ? t.left[i] : t.right[i]) : v <= t.threshold[i] ? t.left[i] : t.right[i];
  }
  throw new Error("gbdt: tree walk exceeded node count — malformed artifact");
}

// Raw score = plain sum of leaf values. LightGBM folds boost-from-average into
// the first tree rather than carrying a separate init score, so there is
// deliberately no bias term added here; parity would catch it immediately if
// that ever stopped being true.
export function rawScore(b: Booster, x: number[]): number {
  let s = 0;
  for (const t of b.trees) s += walk(t, x);
  return s;
}

// Poisson training means the model works in log space, so the link is exp().
// Counts cannot be negative and a runaway exponent would poison the simulation
// downstream, so the result is clamped to a sane rate — a station cannot see
// hundreds of departures an hour from a 19-dock rack.
const MAX_RATE = 200;

export function predict(b: Booster, x: number[]): number {
  const raw = rawScore(b, x);
  if (b.objective === "regression") return Math.max(0, raw);
  const v = Math.exp(raw);
  return Number.isFinite(v) ? Math.min(v, MAX_RATE) : 0;
}

// The GLM arm: same encoded feature vector, same link, ~40 coefficients instead
// of a few thousand trees. Its only job is to answer whether gradient boosting
// is earning its complexity on this problem.
export function predictLinear(intercept: number, coef: number[], x: number[]): number {
  let s = intercept;
  for (let i = 0; i < coef.length; i++) s += coef[i] * x[i];
  const v = Math.exp(s);
  return Number.isFinite(v) ? Math.min(v, MAX_RATE) : 0;
}
