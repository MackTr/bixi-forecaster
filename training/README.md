# training/ — offline only

Nothing in here ships. Python cannot run in a Cloudflare Worker, so this package
exists to produce exactly two files:

| file | consumed by |
|---|---|
| `out/artifact.json` | uploaded to D1, walked at inference by `src/gbdt.ts` |
| `out/golden.json` | `scripts/parity.ts`, the gate that must pass before activation |

Everything under `data/` and `out/` is gitignored and rebuildable from scratch.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install duckdb lightgbm numpy pandas scikit-learn
```

## Pipeline

Run in order. Each step is idempotent and skips work that already exists, so
re-running is cheap.

```sh
.venv/bin/python -m bixi_train.opendata    # ~8 min  2.8 GB/season -> projected TSV
.venv/bin/python -m bixi_train.weather     # ~2 s    one Open-Meteo archive call
.venv/bin/python -m bixi_train.panel       # ~15 s   dense (station, date, hour) grid
.venv/bin/python -m bixi_train.stations    # ~2 s    per-station profiles
.venv/bin/python -m bixi_train.backtest    # GATE 1: the panel matches the day-level oracle
.venv/bin/python -m bixi_train.train       # ~20 min transfer test, learning curve, held-out scores
.venv/bin/python -m bixi_train.export      # artifact + golden vectors
cd .. && npm run parity                    # GATE 2: TypeScript reproduces Python to 1e-9
```

`opendata` needs `data/2024.zip` and `data/2025.zip`; it downloads them if absent
(~800 MB total). They are only ever streamed, never extracted — decompressed they
are 2.8 GB each.

## The two gates

**`backtest.py`** checks the hourly rebuild against the day-level matrix measured
earlier (`data/oracle_daily.csv`). The decisive test is paired equality: summing
the new panel over 6–11am must reproduce the old matrix *exactly*, station-day for
station-day, across the ~306k rows they share. It also requires the November tail
to **differ**, because the old pass assumed a blanket UTC−4 and the season runs
~12 days past the end of DST.

**`scripts/parity.ts`** checks that `src/features.ts` + `src/gbdt.ts` reproduce
Python's numbers from *raw* inputs — a date string and millimetres of rain, not a
pre-encoded vector. The encoder is where the bugs live; handing TypeScript an
already-encoded vector would test only the tree walk.

## Why the model predicts a shape, not a count

Station 345 grows ~+14%/year and open data lags a quarter, so the newest trips
available for training are already stale. A tree given a raw `year` feature
learns a step function and extrapolates it flat. So:

```
count ~ Poisson( level_station × shape(hour, weather, day, station-kind) )
```

`level_station` enters training as the Poisson **offset** (`init_score = log
level`) and never appears in the trees. LightGBM's `predict(raw_score=True)`
excludes the init score, so Python's raw output is `sum(trees)` — exactly what
`gbdt.ts` computes, with no bias term to reconcile. `src/demand.ts` refits that
one scalar nightly against the last 28 observed mornings, which keeps the trend
outside the model, where it can be extrapolated honestly.

## Constraints worth knowing before changing anything

- **No categorical features.** LightGBM encodes categorical splits as set
  membership; `src/gbdt.ts` only knows `x[f] <= threshold`. Declaring `dow` or
  `hour` categorical trains a model the Worker cannot evaluate.
- **Bucket edges live in the artifact**, not in TypeScript. If they were written
  down in both places, a retrain that moved an edge would produce a Worker that
  encodes its inputs differently than the model was trained on, silently.
- **Missing-value routing is translated, not copied.** See the docstring in
  `export.py`: this panel has no missing values, so every node comes out as
  `missing_type="None"` (LightGBM treats NaN as 0.0) while the Worker does
  sometimes hit missing weather. `defaultLeft` is computed to reproduce that.
- **Station names are the only join key.** Open data has no station id column.
