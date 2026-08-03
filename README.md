# bixi-forecaster

An ML run-out predictor for BIXI station 345 (`Regina / de Verdun`, 19 docks),
built to be **raced side by side** against `bixi-predictor`'s Gaussian model
rather than to replace it.

Three repos, three jobs:

| repo | holds |
|---|---|
| `bixi-monitor` | facts — GBFS observations, and the dashboard |
| `bixi-predictor` | beliefs — a similarity-weighted Gaussian model over ~30 days of the station's own history |
| **`bixi-forecaster`** | a second engine with different failure modes, a blend of the two, and the scoreboard that grades all three |

`bixi-monitor` and `bixi-predictor` are **not modified** by this project.
`src/model.ts` is deliberately not vendored — the control arm is reached over
HTTP, so it is the real deployed model rather than a copy that could drift.

## Status

**Shadow mode began 2026-08-01.** Deployed, `gbdt-20260730-2e733b58` active, all
three arms writing nightly at 22:05 America/Toronto. Notifying nobody.

- [x] Worker skeleton, hourly facts, frozen weather, variant-keyed schema
- [x] `training/` — DuckDB panel, LightGBM, time-blocked CV, learning curve
- [x] `export.py` + `gbdt.ts` + parity gate — **passing at 5.6e-17**
- [x] Artifact upload/activation with a verified SHA chain
- [x] Three arms settled: `gaussian`, `ml`, `blend` — the `glm` arm was cut
- [x] Deployed + first live nightly run — first target date 2026-08-01
- [ ] 40 paired nights, then read `/api/v1/compare` against the pre-registered rule

The first night is not a result. See
[docs/model.md](docs/model.md#the-decision-rule) for what would count as one, and
why a scoreboard read after mid-September is close to a prediction of a null.

## Why it works this way

**Python cannot run in a Cloudflare Worker.** Training is offline; inference must
execute in TypeScript. LightGBM trees are exported as flat numeric arrays and
walked by a ~40-line dependency-free evaluator.

```
Python (offline)                          Worker (nightly)
  DuckDB panel → LightGBM  →  artifact.json  →  D1 → gbdt.ts walks trees → λ(h), μ(h)
                              + golden.json  →  parity gate blocks activation
```

ONNX Runtime Web was evaluated and rejected: Workers cannot instantiate WASM from
bytes fetched at runtime, and ORT's bundle would dwarf the size budget for a model
a tree walk evaluates *exactly* rather than approximately.

The artifact lives in a **versioned D1 table**, not the bundle, so retraining and
rollback are database writes rather than redeploys — and the exact bytes behind
any past prediction stay recoverable.

Two stages, because trip data has no inventory column and can therefore only teach
demand. See **[docs/model.md](docs/model.md)** for the model, its biases, and —
importantly — the pre-registered criteria the A/B will be judged by.

## Layout

```
src/                    Worker — zero runtime dependencies
  worker.ts             entry; DST-proof cron guard
  pipeline.ts           nightly orchestration, per-step error isolation
  sync.ts monitor.ts    hourly digest from bixi-monitor
  weather.ts            Open-Meteo; frozen f_* forecast + a_* actual columns
  features.ts           the encoder — the most dangerous file here
  gbdt.ts artifact.ts   tree walk; versioned, chunked, SHA-256-verified storage
  demand.ts             stage 1 + nightly level calibration
  simulate.ts           stage 2 Monte Carlo
  rebalance.ts          empirical truck priors
  compare.ts            variant mirroring, blend gate, scoreboard
  api.ts                /api/v1 router
  tz.ts holidays.ts     copied verbatim from bixi-predictor
training/               OFFLINE ONLY — see training/README.md
scripts/                parity.ts (the gate), upload-artifact.ts
docs/model.md           the model, and how it will be judged
docs/training.md        how the live artifact was trained, and when
```

## Nightly pipeline

Crons `["5 2 * * *", "5 3 * * *"]`. 10pm Montreal is 02:00 UTC under EDT and 03:00
under EST, so **both fire year-round and only the one landing at local hour 22
does work** — DST-proof without a timezone-aware scheduler. The `:05` offset puts
this run five minutes after `bixi-predictor`'s, so its prediction exists when the
mirror step fetches it.

```
sync-hourly → weather-actuals → weather-forecast (frozen) → rebalance-profile
→ finalize (all variants) → predict-ml → mirror-gaussian → predict-blend
```

Every step is individually try/caught. A monitor outage at 10pm still yields
predictions from existing facts, and a missed night self-heals on the next one via
the 14-day sync lookback. Crucially, each of the three arms is its own step: a
failure in one must not deny the others a row, or the scoreboard silently biases
toward whichever model *fails* least rather than whichever *predicts* best.

`predict-blend` runs last because the gate reads the two rows above it. It
weights them by the Gaussian's own published confidence, averages their times,
and takes its window from the **mixture** of the two beliefs rather than from the
average of their endpoints — which would produce an interval containing neither.
See [docs/model.md](docs/model.md#how-the-blend-composes-what-it-weights).

**Write volume is the one real regression from `bixi-predictor`.** Hourly
granularity means ~336 `hourly_facts` + ~350 `weather_hourly` rows per night versus
~14. Every write goes through `env.DB.batch([...])` or a multi-row INSERT; a
per-row `await .run()` loop will not survive the free-tier budget.

## API

```
GET  /api/v1/health
GET  /api/v1/compare?days=            the scoreboard — paired MAE, coverage, Brier, sign test
GET  /api/v1/stations/345/prediction  ?variant=&date=&curve=1
GET  /api/v1/stations/345/predictions ?days=&all=1   all three variants side by side

POST /api/v1/admin/backfill?days=&force=1            (Bearer ADMIN_TOKEN)
POST /api/v1/admin/run
POST /api/v1/admin/replay?date=[&force=1]            recompute ml + blend from frozen f_*, same seed
                                                     409s on an already-finalized night unless forced
GET  /api/v1/admin/model
POST /api/v1/admin/model/{upload,parity-passed,activate}?version=
```

The prediction response is deliberately shape-compatible with
`bixi-predictor`'s, so the monitor dashboard consumes either service unchanged.

There is **no push here.** Notifications stay in `bixi-predictor` for the whole
shadow period — two workers notifying the same subscriptions would confound the
comparison this service exists to run.

## Development

```sh
npm install
npm run db:migrate:local
npm run dev                # wrangler dev on :8789, --test-scheduled
npm run typecheck          # tsc for src/ (Workers types) and scripts/ (+ Node)
npm run parity             # the gate
npm run scoreboard         # read the race — defaults to production
```

`npm run scoreboard` is the intended way to read the experiment. It prints every
arm's MAE, signed bias, window coverage and Brier, then checks the four
pre-registered criteria in [docs/model.md](docs/model.md#the-decision-rule) one
by one and says plainly whether a challenger has beaten the control. It computes
no skill numbers of its own — all of them come from `/api/v1/compare`, so there
is only ever one definition of an error — and it refuses to declare a winner on
an MAE gap the sample cannot resolve. `--days`, `--base` and `--json` are the
only flags.

Retraining is offline and documented in [training/README.md](training/README.md).
To ship a new artifact:

```sh
npm run upload-artifact -- --base https://bixi-forecaster.bixi.workers.dev --activate
```

That runs the parity gate against the exact bytes it uploads, requires the SHA-256
D1 computed to match its own, and only then records parity and activates. **It
should be the only thing that ever calls `/admin/model/parity-passed`** — that
endpoint cannot verify anything itself, so a hand-run curl can stamp an artifact
nobody tested. Omit `--activate` to upload and verify without going live; add
`--dry-run` to run the gate and touch nothing.

## First deploy

```sh
wrangler d1 create bixi_forecaster     # paste database_id into wrangler.toml
npm run db:migrate
wrangler secret put ADMIN_TOKEN
wrangler deploy
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" .../api/v1/admin/backfill?days=14
npm run upload-artifact -- --base https://... --activate
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" .../api/v1/admin/run
```

Backfill must precede the artifact upload: level calibration needs 12+ usable
morning observations or it silently falls back to the artifact's training-time
level.

## Things that will bite you

- **Dates must flow from `localToday()`/`addDays()`, never `toISOString()`.** The
  cron fires at 02:05/03:05 UTC, when the UTC calendar date is already *tomorrow*
  in Montreal terms.
- **No categorical features.** LightGBM encodes categorical splits as set
  membership; `gbdt.ts` only knows `x[f] <= threshold`. Declaring `dow` or `hour`
  categorical trains a model the Worker cannot evaluate.
- **Never round leaf values when exporting.** Rounding to 1e-9 random-walks across
  600 summed leaves and fails parity on 136/453 rows. Thresholds may be rounded —
  they only decide comparisons.
- **Don't raise the 600-tree cap** without re-checking station 345's column. More
  capacity improves network deviance and makes 345 *worse*.
- **`f_*` weather columns are written once and never overwritten.** That is what
  makes replay honest and train/serve skew measurable.
