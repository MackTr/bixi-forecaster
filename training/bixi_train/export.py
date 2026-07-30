"""The bridge: LightGBM in Python -> flat numeric arrays src/gbdt.ts can walk.

Two files come out of here. The artifact is the model. The golden vectors are the
reason anyone should believe the model means the same thing on the other side.

**Missing-value routing is the subtle part, and it is where a silent parity bug
would live.** src/gbdt.ts knows exactly one rule: NaN goes left if `defaultLeft`,
otherwise right. LightGBM has three regimes:

  * `missing_type = "NaN"` -- NaN routes by the node's `default_left`. Maps over
    directly.
  * `missing_type = "None"` -- LightGBM converts NaN to 0.0 and compares it to
    the threshold like any other value. This panel has no missing values at all,
    so EVERY node comes out this way. Translating `default_left` literally here
    would be wrong: the Worker does hit missing weather occasionally, and the two
    sides would then disagree on exactly the rows nobody is watching.
  * `missing_type = "Zero"` -- real zeros are also treated as missing, which
    gbdt.ts cannot express. It cannot arise under `zero_as_missing=false`, and
    the exporter refuses to continue if it ever does.

So `defaultLeft` is not copied, it is COMPUTED: for "None" nodes it becomes
`0.0 <= threshold`, which reproduces LightGBM's own NaN-as-zero behaviour inside
the single boolean the TypeScript already understands. The golden vectors include
rows with missing weather specifically to exercise this.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from . import features, paths, stations, train
from .features import FEATURE_NAMES
from .panel import connect

ARTIFACT = paths.OUT / "artifact.json"
GOLDEN = paths.OUT / "golden.json"

# Thresholds are rounded; leaf values are NOT, and the asymmetry is the whole
# point.
#
# A threshold only has to decide a comparison. LightGBM puts them at bin
# midpoints, nowhere near a real feature value, so nine decimals is far more
# resolution than any decision needs and it buys back a chunk of the JSON a
# Worker parses on every cold start.
#
# A leaf value gets SUMMED, 600 times per prediction. Rounding each one to 1e-9
# looks harmless and is not: the errors random-walk into ~1e-9 of relative error
# in the final rate, which is exactly the parity tolerance. Rounding them was
# measured pushing 136 of 453 golden rows past the gate. Left exact, Python and
# TypeScript sum identical doubles in identical order and agree to the last bit.
ROUND_THRESHOLD = 9
ROUND_VALUE = None


def flatten_tree(struct: dict) -> dict:
    """One LightGBM tree_structure -> six parallel arrays, node 0 = root."""
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []
    default_left: list[int] = []

    def add(node: dict) -> int:
        idx = len(feature)
        if "leaf_value" in node and "split_index" not in node:
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            v = float(node["leaf_value"])
            value.append(round(v, ROUND_VALUE) if ROUND_VALUE is not None else v)
            default_left.append(0)
            return idx

        dtype = node.get("decision_type", "<=")
        if dtype != "<=":
            # A categorical split encodes set membership, which the flat-array
            # evaluator has no representation for.
            raise SystemExit(f"unsupported decision_type {dtype!r} -- no categorical features may be declared")
        mtype = node.get("missing_type", "None")
        if mtype == "Zero":
            raise SystemExit("missing_type='Zero' cannot be represented by gbdt.ts -- set zero_as_missing=false")

        thr = round(float(node["threshold"]), ROUND_THRESHOLD)
        # See the module docstring: for "None" nodes LightGBM treats NaN as 0.0,
        # so the equivalent single-boolean rule is whether 0.0 goes left.
        dleft = bool(node.get("default_left", False)) if mtype == "NaN" else (0.0 <= thr)

        feature.append(int(node["split_feature"]))
        threshold.append(thr)
        left.append(-1)
        right.append(-1)
        value.append(0.0)
        default_left.append(1 if dleft else 0)

        li = add(node["left_child"])
        ri = add(node["right_child"])
        left[idx], right[idx] = li, ri
        return idx

    add(struct)
    return {
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "value": value,
        "defaultLeft": default_left,
    }


def flatten_booster(booster) -> dict:
    dump = booster.dump_model()
    # gbdt.ts applies exp() unconditionally for the poisson objective; a booster
    # trained under any other link would be evaluated through the wrong one.
    obj = (dump.get("objective") or "").split()
    if not obj or obj[0] != "poisson":
        raise SystemExit(f"expected a poisson booster, got {dump.get('objective')!r}")
    # The feature order baked into the model must be the order the encoder emits;
    # a mismatch here is the failure the whole contract exists to prevent.
    names = dump.get("feature_names") or []
    if list(names) != list(FEATURE_NAMES):
        raise SystemExit(f"model feature order {names} != encoder order {list(FEATURE_NAMES)}")
    return {"objective": "poisson", "trees": [flatten_tree(t["tree_structure"]) for t in dump["tree_info"]]}


def golden_rows(df: pd.DataFrame, enc: dict, prof: dict, n: int, seed: int) -> pd.DataFrame:
    """Raw rows for the parity gate, chosen to cover the encoder's edges.

    Sampling uniformly would produce 500 dry summer weekday mornings and prove
    almost nothing. The gate is only as good as the corners it visits, so rain
    buckets, holidays, weekends, the DST shoulder and missing weather are all
    forced in explicitly.
    """
    rng = np.random.default_rng(seed)
    picks = [
        df[df["precip_bucket"] == 1],
        df[df["precip_bucket"] == 2],
        df[df["night_precip_bucket"] == 2],
        df[df["is_holiday"] == 1],
        df[df["is_weekend"] == 1],
        df[df["hour"].isin([0, 6, 8, 12, 22, 23])],
        # November, where the EST/EDT shoulder lives.
        df[pd.to_datetime(df["date"]).dt.month == 11],
        df,
    ]
    per = max(1, n // (len(picks) + 1))
    idx: list[int] = []
    for p in picks:
        if len(p):
            idx.extend(rng.choice(p.index.to_numpy(), size=min(per, len(p)), replace=False).tolist())
    return df.loc[sorted(set(idx))].head(n)


def build_golden(df: pd.DataFrame, boosters: dict, enc: dict, prof: dict, n: int, seed: int) -> dict:
    sample = golden_rows(df, enc, prof, n, seed)
    rows = []
    for _, r in sample.iterrows():
        raw = {
            "date": r["date_str"],
            "hour": int(r["hour"]),
            "tempC": float(r["temp_c"]),
            "precipMm": float(r["precip_mm"]),
            "precipProb": None,
            "windKmh": float(r["wind_kmh"]),
            "dewC": float(r["dew_c"]),
            "nightPrecipMm": float(r["night_mm"]),
        }
        rows.append(raw)

    # Serving-only paths the training panel can never contain, appended by hand.
    # precip_prob does not exist in the archive API, and missing weather is a
    # thing the Worker hits but the panel filtered out -- both must still be
    # proven identical across the two encoders.
    base = rows[0]
    for prob in (0.0, 49.0, 50.0, 51.0, 100.0):
        rows.append({**base, "precipMm": 0.0, "precipProb": prob})
    for prob in (50.0, 100.0):
        # Promotion must NOT fire on an already-wet bucket, nor on the night one.
        rows.append({**base, "precipMm": 9.0, "precipProb": prob})
    for field in ("tempC", "precipMm", "windKmh", "dewC", "nightPrecipMm"):
        rows.append({**base, field: None})
    rows.append({k: (None if k not in ("date", "hour") else v) for k, v in base.items()})

    out = []
    for raw in rows:
        x = features.encode_one(raw, enc, prof)
        xa = np.asarray([x], dtype=np.float64)
        out.append(
            {
                "raw": raw,
                "encoded": [None if np.isnan(v) else v for v in x],
                "departures": float(np.exp(boosters["dep"].predict(xa, raw_score=True)[0])),
                "arrivals": float(np.exp(boosters["arr"].predict(xa, raw_score=True)[0])),
            }
        )
    return {"features": list(FEATURE_NAMES), "encoders": enc, "station": prof, "rows": out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=int, default=500)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--rounds", type=int, default=0, help="fixed boosting rounds (0 = early stopping)")
    a = ap.parse_args(argv)
    paths.ensure_dirs()

    df = features.build_matrix()
    features.check_paths_agree(df, features.DEFAULT_ENCODERS)
    print(f"[export] matrix {len(df):,} rows")

    with connect() as con:
        prof_df = stations.compute(con)
    row = prof_df[prof_df["name"] == paths.STATION_345]
    if row.empty:
        raise SystemExit(f"no profile for {paths.STATION_345!r}")
    r = row.iloc[0]
    station = {
        "morningShare": float(r["morning_share"]),
        "logLevel": float(r["log_level"]),
        "netBalance": float(r["net_balance"]),
    }

    # The shipped model trains on everything. The held-out scores that justify it
    # come from train.py; this is the fit that goes to production.
    boosters = {}
    for target in ("dep", "arr"):
        b = train.fit(df, target, rounds=a.rounds or train.NUM_ROUNDS)
        boosters[target] = b
        print(f"[export] {target}: {b.num_trees()} trees")

    artifact = {
        "version": "",  # filled from the content hash below
        "kind": "gbdt",
        "createdAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "features": list(FEATURE_NAMES),
        "encoders": features.DEFAULT_ENCODERS,
        "station": station,
        "departures": flatten_booster(boosters["dep"]),
        "arrivals": flatten_booster(boosters["arr"]),
    }
    body = json.dumps({k: v for k, v in artifact.items() if k != "version"}, sort_keys=True)
    # Version derived from the bytes, so the same panel and seed produce the same
    # version and a changed model can never quietly reuse one.
    artifact["version"] = "gbdt-" + dt.date.today().strftime("%Y%m%d") + "-" + hashlib.sha256(body.encode()).hexdigest()[:8]

    ARTIFACT.write_text(json.dumps(artifact, separators=(",", ":")))
    nodes = sum(len(t["feature"]) for k in ("departures", "arrivals") for t in artifact[k]["trees"])
    print(f"[export] {ARTIFACT} — {ARTIFACT.stat().st_size / 1e6:.1f} MB, {nodes:,} nodes, version {artifact['version']}")

    golden = build_golden(df, boosters, features.DEFAULT_ENCODERS, station, a.golden, a.seed)
    golden["version"] = artifact["version"]
    GOLDEN.write_text(json.dumps(golden, separators=(",", ":")))
    print(f"[export] {GOLDEN} — {len(golden['rows'])} golden rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
