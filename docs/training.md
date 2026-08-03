# How the model was trained

Three documents, three jobs. [`docs/model.md`](model.md) explains *what* the model
is and why it is shaped that way. [`training/README.md`](../training/README.md) is
the runbook for re-running the pipeline. **This file is the record of the specific
run that produced the artifact now serving predictions** — when it happened, what
went in, what came out, and what it would take to reproduce it.

That record has to live here because it lives nowhere else: `training/data/` and
`training/out/` are both gitignored, so no input to this run and no output of it
is in version control. The model *bytes* are durable — they are in D1, chunked and
SHA-256 verified — but everything upstream of them is not.

---

## The artifact in production

| | |
|---|---|
| **Version** | `gbdt-20260730-2e733b58` |
| **Trained** | **2026-07-30, 18:53 UTC** (training finished) |
| **Exported** | **2026-07-30, 19:12:55 UTC** (`createdAt` in the artifact) |
| **Parity passed** | 2026-07-30, max abs diff 5.6e-17 across 453 golden rows |
| **Activated in production** | 2026-08-01, first live prediction for target date 2026-08-01 |
| **Size** | 4,521,735 bytes → 18 chunks in D1 |
| **Contents** | 2 boosters × 600 trees = 1,200 trees, 150,000 nodes, 75,600 leaves |
| **Objective** | `poisson`, both boosters |
| **Features** | 13, order fixed — see [model.md](model.md#the-13-features) |
| **Encoder edges** | `precipDryMaxMm 0.5`, `precipWetMinMm 4.0`, `probBumpThreshold 50.0` |
| **Station 345 profile** | `morningShare 0.2512`, `logLevel 2.697663` (= **14.84** morning departures/day), `netBalance -0.4700` |

The version string is **content-addressed**: `gbdt-<date>-<first 8 hex of the
SHA-256 of the artifact body with the version field removed>`
([export.py:253](../training/bixi_train/export.py)). A changed model therefore
cannot quietly reuse a version string — if two artifacts share a version they are
byte-identical, and if they differ the version differs.

`logLevel` is a *training-time* level and is not what serves predictions. It is
the fallback. Every night `calibrateLevel()` refits that one scalar against the
last 28 observed mornings, and only falls back to 14.84 when fewer than 12 usable
points exist. See [model.md](model.md#shape-not-level).

---

## Timeline

| when (local, EDT) | what |
|---|---|
| 2026-07-27 19:06 | 2024 + 2025 open-data zips downloaded (~800 MB) |
| 2026-07-27 20:09–20:13 | streamed and projected to `trips_2025.tsv` (360 MB), `trips_2024.tsv` (334 MB) |
| 2026-07-27 22:40 | Open-Meteo archive pulled — one call, 731 days, 657 KB |
| 2026-07-30 14:32 | `oracle_daily.csv` rebuilt — the day-level sanity oracle |
| 2026-07-30 14:53 | **training finished**, `train_report.json` written |
| 2026-07-30 15:12:55 | **`artifact.json` + `golden.json` exported** |
| 2026-07-30 15:13 | `bixi.duckdb` final write (126 MB) |
| 2026-07-30 ~15:20 | parity gate passed under Node |
| 2026-08-01 ~00:00 | uploaded to production D1 and activated |

Roughly three days elapsed, but only ~25 minutes of it was compute. The gap is
the falsification work described under [Validation](#how-it-was-validated) — the
transfer test and learning curve were run and read before the final model was
accepted, not after.

---

## The environment

```
Python      3.9.6
lightgbm    4.6.0
duckdb      1.4.5
numpy       2.0.2
pandas      2.3.3
scikit-learn 1.6.1
```

Pinned nowhere. There is no lockfile for `training/.venv` — the setup line in
`training/README.md` installs whatever is current. Reproducing this artifact
byte-for-byte on a fresh machine would require these versions; LightGBM's tree
construction is not guaranteed stable across minor releases.

Determinism *within* a version is explicit: `seed: 20260727` and
`deterministic: true` are set in the training params, and `num_threads: 0`
(all cores) is safe because `deterministic` forces reproducible histogram
construction regardless of thread count.

---

## The data that went in

| source | detail |
|---|---|
| BIXI open data 2024 | 13.3M trips, 1,094 stations |
| BIXI open data 2025 | 14.2M trips, 1,277 stations |
| Open-Meteo archive | hourly temp / precip / wind / dew, 2024-01-01 → 2026-01-01, zero nulls |
| **Resulting panel** | **5,190,555 rows** at (station, date, hour) |

Schema for 2024 and 2025 was verified byte-identical. There is **no station id**
(the join key is the station *name* — 345 is `Regina / de Verdun`), **no bike-type
column**, and no trip id. Timestamps are epoch milliseconds UTC.

### Two seasons, deliberately

Not a default. The learning curve was run to decide it:

| fold | 2025 only | 2024+2025 | gain |
|---|---|---|---|
| summer | 1.5355 | 1.5053 | **1.97%** |
| autumn | 1.4400 | 1.4054 | **2.40%** |

Season two buys ~2% deviance, which is the plan's own "barely improves"
threshold — so seasons three and four will not pay either. Older seasons also
carry regime shifts that cannot be conditioned away: e-bike fleet growth with no
bike-type column to condition on, network growth (1,094 → 1,277 stations in one
year), and post-pandemic commute normalisation, which contaminates the weekday
morning peak — the single signal this model most depends on.

**Do not ingest 2022/2023.** Their header compatibility is also unverified, and
pre-2022 uses station codes rather than names.

### What survives on disk, and what does not

`training/data/2024.zip` and `2025.zip` are **dangling symlinks** — they pointed
into a session scratchpad that has since been cleaned up. The raw zips are gone.

What remains and is sufficient to rebuild the panel without re-downloading:

```
trips_2024.tsv        334 MB   projected departures/arrivals
trips_2025.tsv        360 MB
stations_2024.tsv     36 KB    per-station profiles
stations_2025.tsv     43 KB
weather_hourly.csv    561 KB
weather_raw.json      675 KB   the untouched Open-Meteo response
bixi.duckdb           126 MB   the built panel
oracle_daily.csv      29 MB    the day-level sanity oracle
```

If those are lost too, `opendata.py` re-downloads from the URLs in
`training/bixi_train/opendata.py`. Budget ~8 minutes for the download and stream.

---

## The pipeline, step by step

```sh
.venv/bin/python -m bixi_train.opendata    # ~8 min   stream 2.8 GB/season from the zip
.venv/bin/python -m bixi_train.weather     # ~2 s     one Open-Meteo archive call
.venv/bin/python -m bixi_train.panel       # ~15 s    dense (station, date, hour) grid
.venv/bin/python -m bixi_train.stations    # ~2 s     per-station profiles
.venv/bin/python -m bixi_train.backtest    # GATE 1   panel matches the day-level oracle
.venv/bin/python -m bixi_train.train       # ~20 min  transfer test, learning curve, scores
.venv/bin/python -m bixi_train.export      # ~3 min   artifact + golden vectors
cd .. && npm run parity                    # GATE 2   TypeScript reproduces Python
```

Two decisions inside `opendata` are load-bearing and were arrived at the hard way:

- **awk projects, DuckDB aggregates.** Aggregating in awk at hourly granularity
  builds an ~8M-key associative array and becomes the bottleneck. Having awk emit
  one short line per trip and letting DuckDB do the `GROUP BY` takes 4–5 min per
  season instead.
- **The UTC offset is derived from the tz database, not hardcoded.** A blanket
  `-14400` is exact from March to early November and *wrong* for the last ~12 days
  of the season, when BIXI still runs but EST has started. 72% of November
  station-days come out different under a blanket offset.

---

## Hyperparameters

```json
{
  "objective": "poisson",       "learning_rate": 0.05,
  "num_leaves": 63,             "min_data_in_leaf": 500,
  "feature_fraction": 0.9,      "bagging_fraction": 0.8,
  "bagging_freq": 1,            "lambda_l2": 1.0,
  "max_bin": 255,               "num_threads": 0,
  "seed": 20260727,             "deterministic": true,
  "boost_from_average": false,  "verbose": -1
}
```

Two of these are decisions rather than defaults:

**`boost_from_average: false`** — the station level enters as the Poisson offset
(`init_score = log level`) and must not be re-learned as a bias. With this off,
`predict(raw_score=True)` returns exactly `sum(trees)`, which is precisely what
`gbdt.ts` computes. There is no bias term for the two implementations to disagree
about, and the parity gate verifies that rather than assuming it.

**600 trees is a cap.** More capacity keeps improving network-wide deviance
(1.5053 → 1.4596) while making station 345 **worse** (3.39 → 3.65) at 6× the
artifact size. The network metric and the metric this service is actually judged
on point in opposite directions past this point. Do not raise it without
re-checking the station-level column.

**No categorical features**, at any setting. LightGBM encodes categorical splits
as set membership; `gbdt.ts` only knows `x[f] <= threshold`. Declaring `dow` or
`hour` categorical would train a model the Worker cannot evaluate — and it would
fail at *parity*, not at deploy, which is the right place but an expensive
surprise.

---

## How it was validated

### Time-blocked, never random

Two forward-chained folds — train on everything before a cutoff, test after it:

| fold | train through | test window |
|---|---|---|
| summer | 2025-08-01 | 2025-08-01 → 2025-09-16 |
| autumn | 2025-09-16 | 2025-09-16 → 2025-11-16 |

A random split would leak: adjacent hours of the same station-day are strongly
correlated, so a shuffled test set is mostly interpolation between rows the model
already saw. Held-out error would look far better than anything achievable at
10pm on a night that has not happened.

### The falsifier that mattered most

The plan's second-largest risk was that network → 345 transfer simply fails, and
the model ends up being an expensive way to say "345's mean" — something
`bixi-predictor` already does better. That was tested directly, with **station 345
held entirely out of training**:

| fold | ML morning MAE | baseline from 345's own history | n |
|---|---|---|---|
| summer | **3.36** | 4.50 | 46 days |
| autumn | 4.84 | 4.97 | 61 days |

Transfer works in summer. In autumn it is a wash, and hourly deviance is actually
*worse* than the baseline.

Training *with* 345 included scores 3.39 — its own ~400 rows add nothing, which
confirms the network-wide design rather than merely permitting it.

**The honest reading: the ML arm's advantage is a summer phenomenon.** Any
scoreboard whose nights fall mostly after mid-September is close to a prediction
of a null result, and [model.md](model.md#what-would-make-the-reading-invalid)
names that as an invalidating condition up front.

### Gate 1 — the panel oracle

`backtest.py` requires the hourly rebuild to reproduce an independently-built
day-level matrix **exactly**, station-day for station-day, when summed over
6–11am. It also requires the November tail to **disagree**, because the old pass
assumed a blanket UTC−4 — if it matched, the new DST handling would not be doing
anything.

An off-by-one hour conversion is invisible in a daily total and survives training,
validation, export and serving. This is the only check that would catch it.

### Gate 2 — Python↔TypeScript parity

453 golden rows of **raw** inputs through both implementations, required to agree
to 1e-9. Achieved **5.6e-17**.

Read the coverage caveat in
[model.md](model.md#scriptsparityts--pythontypescript-agreement) before treating
this as an integrity check — it exercises 38% of departure leaves and 31% of
arrival leaves, and a single wrong leaf in the unvisited majority passes.

---

## Results

Held-out, on the folds above. `deviance` is mean Poisson deviance (lower better);
the baseline is the seasonal-mean predictor.

| fold | target | n | deviance | baseline | MAE | baseline |
|---|---|---|---|---|---|---|
| summer | departures | 582,405 | **1.5053** | 1.9395 | 1.314 | 1.363 |
| summer | arrivals | 582,405 | **1.4966** | 1.8443 | 1.307 | 1.319 |
| autumn | departures | 774,525 | **1.4054** | 1.6303 | 1.127 | 1.256 |
| autumn | arrivals | 774,525 | **1.4156** | 1.6128 | 1.124 | 1.243 |

Deviance improves 12–22% over baseline; MAE barely moves. That is expected and
not a disappointment — MAE on hourly counts averaging ~1.3 is dominated by
Poisson noise no model can remove. Deviance is the metric that reflects whether
the *rates* are right, and the rates are what stage 2 consumes.

**None of these numbers are the thing being tested.** They measure hourly demand
network-wide. What this service is judged on is run-out time at one station, which
is stage 1 composed with stage 2 — and the composition is
[not yet backtested](model.md#known-gaps).

---

## Reproducibility

Given the same code and the same package versions, a re-run reproduces this
artifact: the seed is fixed, `deterministic: true` is set, and the version string
is a content hash, so a byte-identical result is *checkable* rather than assumed —
re-run `export.py` and confirm you get `gbdt-<today>-2e733b58`.

What would change it:

- **Different LightGBM version.** Tree construction is not stable across minor
  releases. This is the most likely cause of an unexpected new hash.
- **Re-downloaded open data.** BIXI restates published seasons occasionally.
- **A moved bucket edge or a reordered feature.** Both ship inside the artifact,
  so both change the hash — and `assertFeatureContract()` will refuse an artifact
  whose feature order disagrees with `features.ts` rather than silently mis-encode.

What is already lost: the raw 2024/2025 zips (see
[above](#what-survives-on-disk-and-what-does-not)). Everything downstream of them
survives, so this is a recoverable gap, not a broken chain.

---

## When to retrain

No schedule. Retraining is a database write, not a redeploy — upload a new
version and activate it — so the cost is low, but every activation **restarts the
A/B window**, because a scoreboard spanning two models measures neither. Every
prediction row stores `model_version` so this is checkable rather than
remembered.

Signals worth acting on:

- **`ratioToTraining` drifting well away from 1.0** over weeks. Published nightly
  in `basis_json`. A slow climb is the +14%/year trend and is *handled* by the
  nightly calibration — that is the design working, not a reason to retrain. A
  sudden jump means something broke.
- **The 2026 season landing in open data** (expect ~Q1 2027 for a full season).
  This is the retrain that matters: it would replace 2024 with 2025+2026 and move
  the training window past the regime this model was fitted in.
- **A measured train/serve skew** large enough to inject into training features —
  the `f_*` / `a_*` columns accumulate that evidence within a season.
- **Not** a bad week. Forty nights is the resolution of this experiment; ten are
  noise.

Whatever the trigger, the sequence is fixed: rebuild, both gates green, then
`npm run upload-artifact -- --activate`, which refuses to record parity unless the
SHA-256 D1 computed matches the bytes the gate ran against.
