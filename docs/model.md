# The model, and how it will be judged

This document exists mostly for one section: [How this will be
judged](#how-this-will-be-judged), written **before** the first night was graded.
A success criterion chosen after you can see the data is not a criterion, and
~40 nights is few enough that the temptation to pick one is real.

---

## Two stages, because trip data has no inventory

BIXI's open trip CSVs record departures and arrivals. They do not record how many
bikes were in the rack. So the model can only learn **demand**, and demand is not
a run-out time — the same 14 departures empty a rack that started at 9 and leave
one that started at 19 half full.

```
stage 1  ML          hourly departure/arrival rates  λ(h), μ(h)
stage 2  simulation  λ, μ + tonight's REAL 10pm count → run-out time + probability + window
```

The join between them is the one number the model does not have to guess:
tonight's actual inventory, observed by `bixi-monitor` an hour before the
prediction runs.

That seeding also makes the thin rebalancing prior survivable. The largest truck
event of the day is the evening sweep, and it is already inside the initial
condition. Only overnight and morning moves have to be modelled.

## Stage 1 — demand

LightGBM, Poisson objective, 600 trees at `lr=0.05`, trained network-wide across
2024+2025 (27.5M trips, ~1,280 stations, 5.19M station-hour rows). Two separate
boosters: departures are censored once a station empties, arrivals are not.

> This section is the *design*. For the run that produced the live artifact —
> dates, package versions, hyperparameters, held-out scores and reproducibility —
> see [docs/training.md](training.md).

**Why network-wide.** Station 345 contributes only ~532 station-days. A model
fitted to those alone is a noisy restatement of the station's own mean, which
`bixi-predictor` already computes better. Transfer was measured with 345 held out
of training entirely:

| fold | morning MAE | baseline from 345's own history | verdict |
|---|---|---|---|
| summer (n=46 days) | **3.36** | 4.50 | transfer works |
| autumn (n=61 days) | 4.84 | 4.97 | a wash — hourly deviance is actually worse |

345-in-training scores 3.39, i.e. its own rows add nothing. **The advantage is a
summer phenomenon**, and that should be remembered when reading any scoreboard
whose nights fall mostly after mid-September.

### The 13 features

`hour`, `dow`, `is_weekend`, `is_holiday`, `month`, `temp_c`, `precip_bucket`,
`wind_kmh`, `dew_c`, `night_precip_bucket`, `st_morning_share`, `st_log_level`,
`st_net_balance`.

Order is a contract — `assertFeatureContract()` refuses an artifact whose list
disagrees, because a shuffled vector of numbers looks fine to the type system.

Rain is **bucketed** (dry / light / wet) rather than passed as millimetres. Two
reasons, and they agree: the measured signal is already bucket-shaped (14.22 /
11.59 / 8.89 mean weekday morning departures, n=243,596), and at 10pm the input is
a *forecast* — a 2mm error barely moves a bucket and moves a continuous feature a
lot. Temperature is weak and non-monotonic (13.31 / 14.50 / 14.71 / 13.69 across
<15 / 15–20 / 20–25 / ≥25 °C), so it is handed over raw for the trees to carve up.

Bucket edges ship **inside the artifact**, not in TypeScript. Written down in both
places, a retrain that moved an edge would produce a Worker that encodes its
inputs differently than the model was trained on, and nothing would fail loudly.

### Shape, not level

Station 345 grows about **+14%/year**, and BIXI's open data lags a quarter — the
newest trips available for training are already stale. A tree handed a raw `year`
feature learns a step function and then extrapolates it *flat*, which is the one
shape guaranteed to be wrong.

So the trees never see a level:

```
count ~ Poisson( level_station × shape(hour, weather, day, station-kind) )
```

`level_station` enters training as the Poisson offset (`init_score = log level`)
and is refit nightly by least squares through the origin against the last 28
observed mornings. Extrapolating a trend is the one thing trees cannot do, and it
is the only thing kept outside them.

Calibration counts only hours the station could actually serve (`empty_minutes ≤
15`). Fitting to censored counts would drag the level down precisely on the
busiest days — the ones this service exists for. Below 12 usable points it falls
back to the artifact's training-time level rather than chase noise, and publishes
`ratioToTraining` every night so the drift is visible.

### Why 600 trees is a cap, not a default

More capacity keeps improving network-wide deviance (1.5053 → 1.4596) while making
station 345 **worse** (3.39 → 3.65) at 6× the artifact size. Do not raise it
without re-checking the station-level column.

## Stage 2 — run-out simulation

150 Monte Carlo paths, 15-minute steps, 22:00 → noon, seeded with the observed
10pm count. Poisson departures and arrivals per step against a hard floor at zero
and a ceiling at capacity (19 docks). Truck visits resolve at hour boundaries from
an empirical `P(visit | dow, hour)` prior — a van shows up, works the rack, and
leaves, and that abruptness is what decides a run-out.

Monte Carlo rather than a closed form because the wanted answer is a
distribution. A censored, path-dependent process has no tidy P25–P75.

The PRNG is seeded from the target date, never the clock, so
`POST /admin/replay?date=` reproduces a night exactly. That is the only way to
prove a change altered the model rather than the dice.

### Two biases that are documented rather than corrected

**Censored demand.** When the rack empties, riders who wanted a bike walked away,
and the trip data never recorded them either. The demand curve is therefore
systematically light in exactly the hours around a run-out. Correcting it would
mean inventing riders, so it stands — but it biases run-out predictions *late*,
and that direction should be checked against the `bias` column in the scoreboard
rather than assumed to be small.

**Rebalancing is an irreducible variance floor.** Trucks move ~13.5 bikes/day at
this station in summer and are invisible in trip data. With ~6 weeks of monitor
observations the prior is thin; it shrinks toward the pooled rate and lets the
Monte Carlo express the resulting uncertainty in the window rather than faking a
point estimate.

## Train/serve skew

Training uses observed precipitation; at 10pm you have a forecast, and the archive
API has no `precipitation_probability`. Three mitigations, in order:

1. Coarse buckets, as above — a bucket survives a 2mm forecast error.
2. `precip_prob` is **excluded from the feature vector** and applied at inference
   as the same deterministic promotion rule `bixi-predictor` uses (dry → light
   above the threshold). The encoder stays identical between train and serve, and
   the two services never disagree about the weather.
3. `weather_hourly` keeps forecast `f_*` and actual `a_*` columns side by side
   **forever**. `bixi-predictor` overwrites its forecast, destroying the record;
   here the skew stays measurable within a season and can be injected into
   training features at the next retrain.

---

## How this will be judged

**Everything in this section is fixed before the first graded night.**

### The comparison

Three variants are written every night, keyed `(target_date, variant)`:

| variant | source | role |
|---|---|---|
| `gaussian` | mirrored over HTTP from `bixi-predictor` | the control |
| `ml` | GBDT demand + Monte Carlo | the challenger |
| `blend` | frozen rule-based gate | the likely winner |

`GET /api/v1/compare?days=` grades all three with one piece of code against one
definition of the actual, on **paired nights only** — nights where both arms
produced a number. An unpaired MAE lets a model look good by abstaining on hard
nights.

A fourth arm, `glm` (Poisson GLM through the same stage 2), was specified and
then **cut before the first graded night**. It asked "is boosting earning its
complexity?" — a question about this implementation rather than about which
model predicts better — and `src/glm.ts` was never written, so it failed
nightly. The variant value survives in the schema and in `Variant`: if it is
ever built, the frozen `f_*` weather columns let it be backfilled into past
nights by replay, and it can be scored then without disturbing this window.

The blend rule is **frozen, not fitted**. With ~40 paired nights a tuned weight
evaluated on those same nights is overfitting. It gates on the Gaussian model's
own published confidence (`fallbackLevel`, `effectiveN`) and records its weights
in `basis_json` nightly, so the rule can be revisited later on data it did not see.

#### How the blend composes what it weights

The gate says how much to trust each arm; composing them is a second decision,
and the two published quantities take different answers.

The **time** is a weighted mean, because averaging point forecasts cancels
independent error. The **window** is the 25th and 75th percentile of the
*mixture* of the two arms' beliefs — each reconstructed as a piecewise-linear
CDF through its published `(p25, median, p75)`, weighted by the gate.

Averaging the window endpoints instead — the rule as first written — produces an
interval containing neither arm when they disagree (Gaussian 8:00–8:40 and ML
10:30–11:30 at .7/.3 average to 8:45–9:31), so the blend would have been
punished on window coverage precisely on the nights blending does work. The
mixture publishes 8:09–10:20 there, and collapses back to exactly 8:00–8:40 when
the arms agree. No new parameter is introduced, so the rule stays frozen.

`basis_json` also records **`mixtureMedian`**, the point estimate the mixture
would have given. Where the arms are far apart the weighted mean lands between
two modes at a time neither model believes, while the mixture median sits inside
whichever arm holds the majority of the weight. Which is better for MAE is an
empirical question ~40 nights cannot settle, so it is stored nightly and graded
by nobody — evaluable later on data it did not see, the same discipline as the
gate itself.

**Both of these were fixed before the first graded night**, which is the only
time such a change is free. Nothing had been deployed and nothing scored.

### The detectable effect

Minimum detectable difference in MAE, paired, power 0.80, α=0.05 two-sided:

| SD of paired diff | n=20 | n=30 | **n=40** | n=50 | n=60 |
|---|---|---|---|---|---|
| 30 min | 18.8 | 15.3 | **13.3** | 11.9 | 10.9 |
| 45 min | 28.2 | 23.0 | **19.9** | 17.8 | 16.3 |
| 56 min | 35.1 | 28.6 | **24.8** | 22.2 | 20.3 |
| 70 min | 43.9 | 35.8 | **31.0** | 27.7 | 25.3 |

Two arms with ~60 min error SD correlated at 0.5–0.7 (they see the same weather
and the same seed inventory) give a paired SD of roughly 45–56 min. So at n≈40
**this experiment can detect a ~20–25 minute difference and nothing smaller.**

A 10-minute improvement is not something ~40 nights can establish. If the
scoreboard shows one, the honest reading is "no detectable difference", not "a
small win".

### Sign test

Wins out of n (ties dropped) needed for two-sided p<0.05 — `signTestTwoSided()`
computes this exactly rather than by normal approximation, because the tail that
matters is the one the approximation gets worst:

| n | wins needed | |
|---|---|---|
| 20 | 15 | 75% |
| 30 | 21 | 70% |
| **40** | **27** | **68%** |
| 50 | 33 | 66% |
| 60 | 39 | 65% |

### The decision rule

A variant is declared better than the control only if **all** of:

1. **n ≥ 40** paired, finalized nights.
2. **MAE improvement ≥ 20 minutes**, i.e. at or above the detectable floor.
3. **Sign test two-sided p < 0.05** — 27+ wins out of 40. This guards against a
   mean driven by a few catastrophic control nights.
4. **Window coverage ≥ 0.5** and **Brier no worse than the control's**. An arm
   that wins on median time while publishing a dishonest window or a miscalibrated
   probability has not won.

Reported alongside, and not to be traded off against each other after the fact:
`bias` (late-vs-early, which MAE hides), `missingNights` per arm, and the
`start_bikes` audit that confirms both arms were seeded with the same inventory.

**If the criteria are not met, the outcome is "no detectable difference."** That
is a real and likely result — and given the transfer table above, a scoreboard
made mostly of autumn nights is close to a prediction of it.

### What would make the reading invalid

- Fewer than 40 paired nights, however good they look.
- Nights concentrated after ~mid-September, where the held-out test says the
  effect does not exist. Report the summer/autumn split.
- Any change to the model, the artifact, or the blend rule mid-window. If that
  happens, the window restarts; `model_version` is stored on every row so this is
  checkable rather than remembered.
- Push notifications moving to this service before the window closes. Two workers
  notifying the same subscriptions would confound the comparison, which is why
  push stays in `bixi-predictor` for the whole shadow period.

---

## What the gates prove — and what they don't

### `scripts/parity.ts` — Python↔TypeScript agreement

453 golden rows of **raw** inputs (a date string and millimetres of rain, not a
pre-encoded vector) go through Python and through `features.ts` + `gbdt.ts`, and
must agree to 1e-9. Current: **5.6e-17**.

Raw rather than pre-encoded is the entire point. The encoder is where the bugs
live — bucket edges, the 0=Sunday convention, holiday dates, the
night-precipitation window, NaN routing. Handing TypeScript a vector Python had
already encoded would test the tree walk, the one part that is nearly impossible
to get wrong.

**This is a path test, not an integrity check.** Measured 2026-07-30: the 453 rows
reach only **14,224 / 37,800 departure leaves (38%)** and **11,565 / 37,800
arrival leaves (31%)**. Demonstrated live — perturbing one leaf of tree 0 by 1e-6
passed with a max diff *identical* to a clean run, because no golden row lands in
that leaf.

That is the right scope for what parity is *for*: encoder bugs are systematic and
shift many rows at once, which is why rounding every leaf value to 1e-9 once
failed 136 of these 453 rows. But **"parity passed" does not mean "artifact
verified"**, and `activateArtifact` should not be read as if it did. To raise
coverage, `export.py` would need to select golden rows by leaf set-cover rather
than at random.

### The SHA-256 chain — artifact integrity

What actually protects the unvisited 62%:

1. `upload-artifact.ts` runs parity against the exact bytes it is about to upload.
2. It requires the digest **D1 computed independently** to equal its own before it
   will call `parity-passed`, then re-reads the metadata to confirm the stamp
   landed on those same bytes.
3. `loadActiveArtifact()` re-verifies the digest on **every cold load**, not just
   at upload — a truncated chunk surfaces as an error rather than as subtly wrong
   predictions.

Between them, transport and storage corruption cannot survive. **What nothing
covers is an error introduced upstream inside `export.py` itself.**

`/admin/model/parity-passed` cannot verify anything on its own — the golden vectors
live on the training machine. It stamps a timestamp and trusts its caller. That is
why `upload-artifact.ts` exists and why it should be the only thing that ever
calls it.

### `training/bixi_train/backtest.py` — the panel oracle

The hourly rebuild must reproduce the independently-built day-level matrix
*exactly*, station-day for station-day, when summed over 6–11am — and the November
tail must **disagree**, because the old pass assumed a blanket UTC−4 while the
season runs ~12 days past the end of DST. An off-by-one hour conversion would
otherwise train, validate, export, and serve a commute peak that is wrong forever.

---

## Known gaps

- **The `glm` arm was cut**, not built — see [The comparison](#the-comparison).
  `predictGlm()` and the artifact's optional `glm` block are still in the tree and
  still correct; only `src/glm.ts` and a pipeline step are missing.
- **Stage 2 is the least validated part.** No inventory exists for 2024–25, only
  the monitor's own weeks. Until it is backtested against those, composition error
  is unquantified — and if it is not clearly under ~56 min MAE, the `ml` arm loses
  for reasons that have nothing to do with the demand model.
- **The truck-domination risk is unfalsified.** If rebalancing controls the
  morning, a better demand model cannot move the metric. Decomposing the monitor's
  observations into burst vs organic would settle it.
- **Two seasons is deliberate.** Season two buys only ~2% deviance (+1.97% summer,
  +2.40% autumn). Older seasons carry regime shifts — e-bike growth with no
  bike-type column, 1,094 → 1,277 stations in a year, post-pandemic commute
  normalization — that contaminate the weekday morning peak, the one signal this
  model most depends on. Do not ingest 2022/2023 on the assumption that more is
  better.
