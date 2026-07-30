"""Stage 1: fit hourly demand, and find out whether it deserves to exist.

The model predicts a dimensionless demand SHAPE, not a count. That is the whole
trick for handling drift. Station 345 grows about +14% a year and open data lags
a quarter, so a tree handed a raw `year` feature learns a step and then
extrapolates it flat -- the one shape guaranteed to be wrong. Instead:

    count ~ Poisson( level_station * shape(hour, weather, day, station-kind) )

`level_station` enters training as the Poisson OFFSET (init_score = log level)
and never appears in the trees at all. At serving time src/demand.ts refits that
single scalar nightly against the last 28 observed mornings, so the trend lives
outside the model, which is the only place it can be extrapolated honestly.

Because LightGBM's `predict(raw_score=True)` does NOT add the init score back,
Python's raw output is exactly `sum(trees)` -- which is exactly what gbdt.ts
computes. exp() of it is the shape. The two sides line up with no bias term to
reconcile, and scripts/parity.ts proves it.

**No categorical features.** LightGBM would encode a categorical split as set
membership, and the flat-array evaluator in src/gbdt.ts only knows how to compare
`x[f] <= threshold`. Declaring `dow` or `hour` categorical would train a better
model that the Worker could not evaluate, and the failure would appear as a
parity mismatch rather than as anything legible. Numeric splits only.

Three questions this file answers, in order of how much they could change the
plan:

1. **Does the network transfer to station 345?** Train with 345 entirely held
   out, then score it. If a network-wide model beats 345's own historical mean,
   the 402k rows are doing real work. If it does not, this model is an expensive
   restatement of what the Gaussian one already knows.
2. **Is a second season worth it?** Train on 2025 alone, then 2024+2025, score on
   the same held-out months. If season two barely helps, seasons three and four
   will not either, and the 2022/2023 ingest (with its unverified schema and
   regime shifts) can be skipped.
3. **Is gradient boosting earning its complexity?** The GLM arm gets the same
   features and the same offset, and rides along in the artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features, paths, stations
from .features import FEATURE_NAMES
from .panel import connect

# Time-blocked, never random. Rows within a day are massively correlated -- a
# random split would put 07:00 in train and 08:00 of the same morning in test and
# report an accuracy that serving could never reproduce.
FOLDS = {
    # The regime the service actually runs in: predicting August from spring and
    # early summer. This is the fold that matters most.
    "summer": {"train_end": "2025-08-01", "test_start": "2025-08-01", "test_end": "2025-09-16"},
    # Autumn, when demand falls and weather turns -- the harder extrapolation.
    "autumn": {"train_end": "2025-09-16", "test_start": "2025-09-16", "test_end": "2025-11-16"},
}

PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbose": -1,
    "num_threads": 0,
    # A reproducible artifact is a debuggable artifact: the same panel must
    # produce the same trees, or a parity failure cannot be bisected.
    "seed": 20260727,
    "deterministic": True,
    # With an offset supplied there is no average to boost from, and leaving this
    # on would add a bias the TypeScript evaluator does not know about.
    "boost_from_average": False,
}
# 600 is a CHOICE, not a default, and early stopping never reaches it -- so this
# cap binds and looks like a bug. It is not. From sweep.py on the summer fold:
#
#   lr=0.05 cap=600    deviance 1.5053   345 morning MAE 3.39    75k nodes
#   lr=0.05 cap=2000   deviance 1.4750   345 morning MAE 3.47   250k nodes
#   lr=0.10 cap=2000   deviance 1.4660   345 morning MAE 3.48   249k nodes
#   lr=0.10 l=127      deviance 1.4596   345 morning MAE 3.65   467k nodes
#
# Network-wide deviance keeps improving with capacity. Station 345 -- the only
# station this service predicts -- gets steadily WORSE. The extra trees are
# spent resolving the other 1,172 stations, and they cost 6x the artifact, which
# a Worker parses on every cold start. Raising this without re-checking the
# station-level column would be trading the objective for the proxy.
NUM_ROUNDS = 600
EARLY_STOP = 40
# Days of the training tail reserved for early stopping. Also time-blocked.
VALID_DAYS = 21


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson deviance -- the loss the model is actually fit to.

    MAE is reported alongside because it is interpretable in bikes, but MAE
    rewards a model that predicts near-zero everywhere on a panel that is 47%
    zeros. Deviance does not.
    """
    mu = np.maximum(mu, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.mean(term - (y - mu)))


def _dataset(df: pd.DataFrame, target: str) -> lgb.Dataset:
    return lgb.Dataset(
        df[FEATURE_NAMES].to_numpy(dtype=np.float64),
        label=df[target].to_numpy(dtype=np.float64),
        init_score=df["st_log_level"].to_numpy(dtype=np.float64),
        feature_name=list(FEATURE_NAMES),
        free_raw_data=False,
    )


def fit(train: pd.DataFrame, target: str, params: dict | None = None, rounds: int = NUM_ROUNDS) -> lgb.Booster:
    """Fit one booster, early-stopping on the tail of the training window."""
    cut = pd.Timestamp(train["date"].max()) - pd.Timedelta(days=VALID_DAYS)
    tr, va = train[train["date"] < cut], train[train["date"] >= cut]
    if len(va) == 0:
        tr, va = train, train
    booster = lgb.train(
        {**PARAMS, **(params or {})},
        _dataset(tr, target),
        num_boost_round=rounds,
        valid_sets=[_dataset(va, target)],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    return booster


def shape_of(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """exp(sum of trees) -- dimensionless, and exactly what gbdt.ts returns."""
    raw = booster.predict(df[FEATURE_NAMES].to_numpy(dtype=np.float64), raw_score=True)
    return np.exp(raw)


def predict_counts(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return shape_of(booster, df) * np.exp(df["st_log_level"].to_numpy(dtype=np.float64))


def station_mean_baseline(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """"Just use this station's own average" -- the thing ML has to beat.

    Keyed on (station, hour, is_weekend), which is roughly the structure the
    Gaussian model already captures from 30 days of its own history. Anything the
    network-wide model cannot beat here, it has no business predicting.
    """
    key = ["station", "hour", "is_weekend"]
    means = train.groupby(key, observed=True)[target].mean().rename("m").reset_index()
    merged = test[key].merge(means, on=key, how="left")
    return merged["m"].fillna(train[target].mean()).to_numpy()


def evaluate(booster: lgb.Booster, train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    y = test[target].to_numpy(dtype=np.float64)
    mu = predict_counts(booster, test)
    base = station_mean_baseline(train, test, target)
    return {
        "n": int(len(test)),
        "deviance": poisson_deviance(y, mu),
        "deviance_baseline": poisson_deviance(y, base),
        "mae": float(np.mean(np.abs(y - mu))),
        "mae_baseline": float(np.mean(np.abs(y - base))),
    }


def morning_total_mae(test: pd.DataFrame, mu: np.ndarray, station: str) -> tuple[float, int]:
    """Daily 6-11am error for one station -- the quantity the service is about.

    Hourly MAE across a 5M-row network is not what anyone experiences; the run-out
    prediction depends on the morning total for a single rack.
    """
    m = test["station"] == station
    if not m.any():
        return float("nan"), 0
    sub = test.loc[m, ["date", "hour", "dep"]].copy()
    sub["mu"] = mu[m.to_numpy()]
    sub = sub[sub["hour"].isin(paths.MORNING_HOURS)]
    g = sub.groupby("date").agg(actual=("dep", "sum"), pred=("mu", "sum"))
    return float(np.mean(np.abs(g["actual"] - g["pred"]))), int(len(g))


def split(df: pd.DataFrame, fold: dict, year_from: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.to_datetime(df["date"])
    train = df[d < pd.Timestamp(fold["train_end"])]
    test = df[(d >= pd.Timestamp(fold["test_start"])) & (d < pd.Timestamp(fold["test_end"]))]
    if year_from is not None:
        train = train[pd.to_datetime(train["date"]).dt.year >= year_from]
    return train, test


# --------------------------------------------------------------------------
# The three questions
# --------------------------------------------------------------------------


def run_learning_curve(df: pd.DataFrame) -> dict:
    """Question 2: does a second season pay for itself?"""
    out: dict = {}
    print("\n=== learning curve: is a second season worth ingesting? ===")
    for fold_name, fold in FOLDS.items():
        out[fold_name] = {}
        for label, year_from in (("2025 only", 2025), ("2024+2025", None)):
            train, test = split(df, fold, year_from)
            t0 = time.time()
            b = fit(train, "dep")
            m = evaluate(b, train, test, "dep")
            m["trees"] = b.num_trees()
            m["train_rows"] = int(len(train))
            m["secs"] = round(time.time() - t0, 1)
            out[fold_name][label] = m
            print(
                f"  {fold_name:7s} {label:10s} train={len(train):>9,} "
                f"deviance={m['deviance']:.4f} mae={m['mae']:.3f} trees={m['trees']}  ({m['secs']}s)"
            )
        a = out[fold_name]["2025 only"]["deviance"]
        b_ = out[fold_name]["2024+2025"]["deviance"]
        gain = (a - b_) / a
        out[fold_name]["gain"] = gain
        print(f"  {fold_name:7s} -> second season changes deviance by {gain:+.2%}")
    return out


def run_transfer(df: pd.DataFrame, station: str = paths.STATION_345) -> dict:
    """Question 1: is the network model more than station 345's own mean?

    345 is removed from training entirely -- not just from the test window, but
    from every row the trees ever see -- so any accuracy on it comes purely from
    other stations that resemble it.
    """
    print(f"\n=== transfer: {station} held out of training entirely ===")
    out: dict = {}
    for fold_name, fold in FOLDS.items():
        train, test = split(df, fold)
        held = train[train["station"] != station]
        test345 = test[test["station"] == station]
        if test345.empty:
            continue
        b = fit(held, "dep")
        mu = predict_counts(b, test345)
        y = test345["dep"].to_numpy(dtype=np.float64)
        base = station_mean_baseline(train[train["station"] == station], test345, "dep")

        sub = test345[["date", "hour", "dep"]].copy()
        sub["mu"], sub["base"] = mu, base
        sub = sub[sub["hour"].isin(paths.MORNING_HOURS)]
        g = sub.groupby("date").agg(actual=("dep", "sum"), pred=("mu", "sum"), base=("base", "sum"))
        m = {
            "n_hours": int(len(test345)),
            "n_days": int(len(g)),
            "hourly_deviance": poisson_deviance(y, mu),
            "hourly_deviance_baseline": poisson_deviance(y, base),
            "morning_mae": float(np.mean(np.abs(g["actual"] - g["pred"]))),
            "morning_mae_baseline": float(np.mean(np.abs(g["actual"] - g["base"]))),
            "morning_mean_actual": float(g["actual"].mean()),
        }
        out[fold_name] = m
        print(
            f"  {fold_name:7s} n={m['n_days']}d  morning MAE {m['morning_mae']:.2f} vs "
            f"own-mean baseline {m['morning_mae_baseline']:.2f} "
            f"(mean actual {m['morning_mean_actual']:.1f} bikes)  "
            f"deviance {m['hourly_deviance']:.4f} vs {m['hourly_deviance_baseline']:.4f}"
        )
    return out


# Question 3 -- the GLM arm -- is deliberately NOT here yet.
#
# src/gbdt.ts's predictLinear walks `coef[i] * x[i]` over the same 13-element
# vector the trees use, which makes the GLM linear in `hour`. A model that
# believes 08:00 is eight times 01:00 loses to trees for reasons that have
# nothing to do with whether boosting earns its complexity, so scoring it would
# produce a number that looks like an answer and is not one. An honest GLM needs
# its own basis expansion (hour dummies -- the ~40 coefficients the plan called
# for), and that is a change to the artifact contract on the TypeScript side.
# It belongs with step 6, where glm.ts is written, not smuggled in here.


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--skip-curve", action="store_true")
    a = ap.parse_args(argv)

    paths.ensure_dirs()
    t0 = time.time()
    df = features.build_matrix()
    features.check_paths_agree(df, features.DEFAULT_ENCODERS)
    print(f"[train] matrix {len(df):,} rows in {time.time() - t0:.0f}s")

    report: dict = {"rows": int(len(df)), "folds": FOLDS, "params": PARAMS}
    if not a.skip_transfer:
        report["transfer"] = run_transfer(df)
    if not a.skip_curve:
        report["learning_curve"] = run_learning_curve(df)

    # Held-out scores for both targets, on the fold the service runs in.
    print("\n=== held-out scores, both targets ===")
    report["scores"] = {}
    for fold_name, fold in FOLDS.items():
        train, test = split(df, fold)
        report["scores"][fold_name] = {}
        for target in ("dep", "arr"):
            b = fit(train, target)
            m = evaluate(b, train, test, target)
            report["scores"][fold_name][target] = m
            print(
                f"  {fold_name:7s} {target}: deviance {m['deviance']:.4f} (baseline {m['deviance_baseline']:.4f}), "
                f"mae {m['mae']:.3f} (baseline {m['mae_baseline']:.3f})"
            )

    out = paths.OUT / "train_report.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[train] report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
