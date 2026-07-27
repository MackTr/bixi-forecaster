-- bixi-forecaster schema. Two things make this different from bixi-predictor's:
-- everything is HOURLY (a 6-11am daily total cannot produce a run-out *time*),
-- and predictions are keyed (target_date, variant) so four models can be graded
-- side by side by one piece of code against one definition of the actual.

-- Per local hour, digested from bixi-monitor's observations. The columns split
-- inventory change into its two causes, because they behave nothing alike:
-- organic flow is Poisson-ish and is what the ML model predicts, while truck
-- visits are large, lumpy, and invisible in BIXI's open trip data. Keeping them
-- in separate columns is what lets the simulation model them separately.
CREATE TABLE hourly_facts (
  date          TEXT NOT NULL,          -- local YYYY-MM-DD
  hour          INTEGER NOT NULL,       -- 0..23 local
  bikes_open    INTEGER,                -- usable bikes at :00 (step-held from last obs at or before)
  bikes_min     INTEGER,                -- minimum seen within the hour
  bikes_close   INTEGER,                -- last value seen within the hour
  empty_minutes INTEGER NOT NULL DEFAULT 0,  -- minutes at 0 usable bikes
  truck_in      INTEGER NOT NULL DEFAULT 0,  -- bikes added by detected truck bursts
  truck_out     INTEGER NOT NULL DEFAULT 0,  -- bikes removed by detected truck bursts
  organic_in    INTEGER NOT NULL DEFAULT 0,  -- arrivals not attributed to a burst
  organic_out   INTEGER NOT NULL DEFAULT 0,  -- departures not attributed to a burst
  obs_count     INTEGER NOT NULL DEFAULT 0,  -- observations landing in the hour
  synced_ts     INTEGER NOT NULL,
  PRIMARY KEY (date, hour)
);

-- One row per local day, deliberately mirroring bixi-predictor.daily_features so
-- both services agree on what "ran out at 7:43" means. If this definition ever
-- drifts from the sibling's, the A/B comparison is measuring the definition
-- rather than the models.
CREATE TABLE daily_facts (
  date            TEXT PRIMARY KEY,     -- local YYYY-MM-DD
  dow             INTEGER NOT NULL,     -- 0=Sun..6=Sat (local)
  is_holiday      INTEGER NOT NULL DEFAULT 0,
  runout_minutes  INTEGER,              -- first usable-bikes >0 -> <=0 transition; NULL = never
  evening_bikes   INTEGER,              -- usable bikes at 22:00 local (step-held)
  evening_swept   INTEGER,              -- bikes trucked out 17:00-22:00 (burst-detected)
  obs_count       INTEGER NOT NULL DEFAULT 0,
  complete        INTEGER NOT NULL DEFAULT 1,  -- 0 while the local day is still running
  synced_ts       INTEGER NOT NULL
);

-- Hourly weather, with forecast and actual SIDE BY SIDE rather than one
-- overwriting the other.
--
-- This is the single most important schema difference from bixi-predictor.
-- There, `weather_daily` overwrites each forecast with the actual once the day
-- settles, which destroys the forecast permanently. Two things become
-- impossible as a result: replaying a past night on the inputs the model
-- actually had (any replay silently leaks actual weather), and measuring
-- train/serve skew at all. Here f_* is written ONCE and never updated, a_*
-- lands alongside it, and the difference between them is a measurable quantity
-- that accumulates on its own from day one.
CREATE TABLE weather_hourly (
  date            TEXT NOT NULL,        -- local YYYY-MM-DD
  hour            INTEGER NOT NULL,     -- 0..23 local; precipitation is the PRECEDING-hour sum
  f_temp_c        REAL,                 -- forecast, frozen at fetch time
  f_precip_mm     REAL,
  f_precip_prob   REAL,                 -- forecast-only: the archive API has no such field
  f_wind_kmh      REAL,
  f_dew_c         REAL,
  f_fetched_ts    INTEGER,              -- when the forecast was frozen (non-NULL = frozen)
  a_temp_c        REAL,                 -- actual, backfilled once the hour is in the past
  a_precip_mm     REAL,
  a_wind_kmh      REAL,
  a_dew_c         REAL,
  a_fetched_ts    INTEGER,
  PRIMARY KEY (date, hour)
);

-- Versioned model artifacts. The trees live in D1 rather than the Worker bundle
-- so a retrain or a rollback is a database write, not a redeploy — and so the
-- exact bytes that produced any past prediction stay recoverable.
--
-- `active` is advisory only: activation is gated on the golden-vector parity
-- test passing under Node against these exact bytes (scripts/parity.ts). An
-- artifact whose TS evaluator disagrees with Python must never go active,
-- because a silent encoder mismatch produces plausible-looking wrong numbers
-- rather than an error.
CREATE TABLE model_artifacts (
  version      TEXT PRIMARY KEY,        -- e.g. "2026-07-27a"
  kind         TEXT NOT NULL,           -- 'gbdt' | 'glm'
  sha256       TEXT NOT NULL,           -- over the reassembled JSON, verified on load
  bytes        INTEGER NOT NULL,
  chunk_count  INTEGER NOT NULL,
  active       INTEGER NOT NULL DEFAULT 0,
  parity_ts    INTEGER,                 -- when parity last passed; NULL = never gate-checked
  notes        TEXT,
  created_ts   INTEGER NOT NULL
);

-- D1 caps a single value's size, and a few thousand trees exceed it. ~256KB
-- chunks reassemble in `idx` order into the JSON verified against sha256 above.
CREATE TABLE model_artifact_chunks (
  version  TEXT NOT NULL,
  idx      INTEGER NOT NULL,
  body     TEXT NOT NULL,
  PRIMARY KEY (version, idx)
);

-- Empirical truck-visit prior, P(visit | dow, hour), built from the monitor's
-- own burst detections. Trip data cannot supply this: BIXI's open data has no
-- inventory, so trucks are invisible in it, yet they move ~13.5 bikes/day at
-- this station in summer.
--
-- With only ~6 weeks of observations this is thin, so consumers shrink it
-- toward the pooled rate and let the Monte Carlo express the resulting variance
-- in the WINDOW rather than faking a confident point estimate. Rebalancing is
-- an irreducible variance floor here, not something the model can predict away.
CREATE TABLE rebalance_profile (
  dow         INTEGER NOT NULL,         -- 0=Sun..6=Sat
  hour        INTEGER NOT NULL,         -- 0..23 local
  p_visit     REAL NOT NULL,            -- P(a truck burst occurs in this cell)
  mean_delta  REAL NOT NULL,            -- mean signed bike change given a visit
  sd_delta    REAL NOT NULL,
  n_days      INTEGER NOT NULL,         -- days observed for this cell — the honesty column
  built_ts    INTEGER NOT NULL,
  PRIMARY KEY (dow, hour)
);

-- One row per (target day, model variant). Four variants race:
--   gaussian  mirrored over HTTP from bixi-predictor — the control
--   ml        GBDT demand + Monte Carlo — the challenger
--   blend     rule-based gate on the Gaussian's own published confidence
--   glm       ~40-coefficient Poisson GLM through the same stage 2 —
--             the "is gradient boosting earning its complexity" arm
--
-- start_bikes is stored per row for an A/B FAIRNESS AUDIT: every variant must
-- have been seeded with the same 10pm inventory. If two variants ever disagree
-- here, the comparison is measuring different starting conditions, not
-- different models.
CREATE TABLE predictions (
  target_date        TEXT NOT NULL,     -- local YYYY-MM-DD
  variant            TEXT NOT NULL CHECK (variant IN ('gaussian','ml','blend','glm')),
  created_ts         INTEGER NOT NULL,
  predicted_minutes  INTEGER,           -- NULL + probability set = "unlikely to run out"
  probability        REAL,              -- NULL = model had nothing to say
  window_early       INTEGER,           -- P25, minutes since local midnight
  window_late        INTEGER,           -- P75
  start_bikes        INTEGER,           -- 10pm seed inventory — fairness audit
  model_version      TEXT,              -- which artifact produced this (NULL for gaussian/blend)
  basis_json         TEXT NOT NULL,     -- explainability; for blend, the frozen gate's weights
  curve_json         TEXT,              -- hourly lambda/mu and the simulated depletion curve
  actual_minutes     INTEGER,           -- backfilled once the target day completes
  error_minutes      INTEGER,           -- predicted - actual, when both non-NULL
  finalized_ts       INTEGER,
  PRIMARY KEY (target_date, variant)
);

-- Scoreboard reads walk variants over a date range.
CREATE INDEX idx_predictions_variant ON predictions (variant, target_date);
