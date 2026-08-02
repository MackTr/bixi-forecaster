// Reads /api/v1/compare and applies the PRE-REGISTERED decision rule from
// docs/model.md mechanically.
//
// This script deliberately computes nothing about model skill. Every number it
// prints comes from the service, which grades all arms with one piece of code
// against one definition of the actual; re-deriving any of it here would create
// a second definition and the two would eventually disagree. What it adds is
// the part a human reading raw JSON gets wrong: checking all four criteria
// instead of the one that looks good, and refusing to call a difference real
// when n is too small for it to be distinguishable from noise.
//
//   node --import ./scripts/ts-ext.mjs scripts/scoreboard.ts [options]
//
//   --base <url>   service base, default https://bixi-forecaster.bixi.workers.dev
//   --days <n>     lookback window, default 60
//   --json         raw response, unformatted

interface VariantScore {
  variant: string;
  n: number;
  mae: number | null;
  bias: number | null;
  windowCoverage: number | null;
  brier: number | null;
  missingNights: number;
}
interface PairedResult {
  a: string;
  b: string;
  n: number;
  maeA: number | null;
  maeB: number | null;
  meanDiff: number | null;
  aWins: number;
  bWins: number;
  ties: number;
  signTestP: number | null;
}
interface Compare {
  window: { from: string; to: string; days: number };
  gradedNights: number;
  scores: VariantScore[];
  paired: PairedResult[];
  seedMismatches: string[];
  interpretation: { detectableEffectMinutes: number; note: string };
}

const CONTROL = "gaussian";

// docs/model.md, "The decision rule". Fixed before the first graded night.
const MIN_NIGHTS = 40;
const MIN_MAE_GAIN = 20; // minutes, i.e. at or above the detectable floor
const MAX_P = 0.05;
const MIN_COVERAGE = 0.5;

// The held-out transfer test found the ML advantage only in summer, so the
// season a scoreboard is made of changes how it should be read. Mid-September
// is where docs/model.md draws that line.
const AUTUMN_FROM = "-09-15";

function parseArgs(argv: string[]) {
  const o = { base: "https://bixi-forecaster.bixi.workers.dev", days: 60, json: false };
  for (let i = 0; i < argv.length; i++) {
    const next = () => {
      const v = argv[++i];
      if (v == null) throw new Error(`${argv[i - 1]} needs a value`);
      return v;
    };
    switch (argv[i]) {
      case "--base": o.base = next().replace(/\/+$/, ""); break;
      case "--days": o.days = parseInt(next(), 10); break;
      case "--json": o.json = true; break;
      default: throw new Error(`unknown flag ${argv[i]}`);
    }
  }
  return o;
}

const pad = (s: string, n: number) => s.padEnd(n);
const padL = (s: string, n: number) => s.padStart(n);
const num = (v: number | null, digits = 1) => (v == null ? "—" : v.toFixed(digits));
const signed = (v: number | null) => (v == null ? "—" : (v > 0 ? "+" : "") + v.toFixed(1));

// Orient a pairwise result so it always reads "challenger vs control".
function versusControl(paired: PairedResult[], challenger: string) {
  const p = paired.find(
    (x) => (x.a === challenger && x.b === CONTROL) || (x.a === CONTROL && x.b === challenger),
  );
  if (!p) return null;
  const flip = p.a === CONTROL;
  const maeChallenger = flip ? p.maeB : p.maeA;
  const maeControl = flip ? p.maeA : p.maeB;
  return {
    n: p.n,
    maeChallenger,
    maeControl,
    // Positive = the challenger is BETTER by this many minutes.
    gain: maeChallenger == null || maeControl == null ? null : Math.round((maeControl - maeChallenger) * 10) / 10,
    wins: flip ? p.bWins : p.aWins,
    losses: flip ? p.aWins : p.bWins,
    ties: p.ties,
    p: p.signTestP,
  };
}

async function main() {
  const o = parseArgs(process.argv.slice(2));
  const res = await fetch(`${o.base}/api/v1/compare?days=${o.days}`);
  if (!res.ok) throw new Error(`compare returned ${res.status} ${await res.text()}`);
  const c = (await res.json()) as Compare;
  if (o.json) return console.log(JSON.stringify(c, null, 2));

  console.log(`\nbixi-forecaster scoreboard    ${c.window.from} → ${c.window.to}  (${c.window.days}d lookback)`);

  if (!c.gradedNights) {
    console.log(`\n  nothing graded yet.\n`);
    console.log(`  A night is graded once its run-out is on the books and a pipeline run has`);
    console.log(`  seen it — so the first row appears after the 22:05 cron following the first`);
    console.log(`  predicted morning. ${MIN_NIGHTS} paired nights are needed before the decision rule`);
    console.log(`  below can be read at all.\n`);
    return;
  }

  // A seed mismatch means the arms were handed different starting inventories,
  // which is a difference in the experiment rather than in the models. Loud and
  // first, because every number underneath it is meaningless if it is non-empty.
  if (c.seedMismatches.length) {
    console.log(`\n  ⚠  SEED MISMATCH on ${c.seedMismatches.length} night(s): ${c.seedMismatches.join(", ")}`);
    console.log(`     The arms did not start from the same 10pm inventory. This invalidates`);
    console.log(`     the comparison until it is explained — do not read past this line.`);
  }

  const seasons = await seasonSplit(o.base, o.days);
  console.log(`graded nights: ${c.gradedNights}${seasons ? `    summer ${seasons.summer} / autumn ${seasons.autumn}` : ""}\n`);

  console.log(`  ${pad("variant", 10)}${padL("n", 4)}${padL("MAE", 8)}${padL("bias", 8)}${padL("window", 9)}${padL("Brier", 8)}${padL("missing", 9)}`);
  for (const s of c.scores) {
    console.log(
      `  ${pad(s.variant, 10)}${padL(String(s.n), 4)}${padL(num(s.mae), 8)}${padL(signed(s.bias), 8)}` +
        `${padL(num(s.windowCoverage, 2), 9)}${padL(num(s.brier, 3), 8)}${padL(String(s.missingNights), 9)}`,
    );
  }
  console.log(`\n  bias is signed: + means the arm predicts LATE, which MAE hides entirely.`);

  const control = c.scores.find((s) => s.variant === CONTROL);
  const challengers = c.scores.filter((s) => s.variant !== CONTROL);
  if (!control) {
    console.log(`\n  no ${CONTROL} rows in this window — nothing to compare against.\n`);
    return;
  }

  console.log(`\npre-registered decision rule — docs/model.md`);
  console.log(`  a challenger beats the control only if ALL FOUR hold\n`);

  for (const s of challengers) {
    const v = versusControl(c.paired, s.variant);
    if (!v) {
      console.log(`  ${s.variant}: no paired nights against ${CONTROL}\n`);
      continue;
    }
    const checks: [boolean, string, string][] = [
      [v.n >= MIN_NIGHTS, `n ≥ ${MIN_NIGHTS} paired nights`, `${v.n}`],
      [(v.gain ?? -1) >= MIN_MAE_GAIN, `MAE improvement ≥ ${MIN_MAE_GAIN} min`, `${signed(v.gain)} min`],
      [(v.p ?? 1) < MAX_P, `sign test p < ${MAX_P}`, `p=${v.p == null ? "—" : v.p.toFixed(4)} (${v.wins}W ${v.losses}L ${v.ties}T)`],
      [
        (s.windowCoverage ?? 0) >= MIN_COVERAGE && (s.brier ?? 1) <= (control.brier ?? 1),
        `window coverage ≥ ${MIN_COVERAGE} and Brier ≤ control`,
        `cov ${num(s.windowCoverage, 2)}, Brier ${num(s.brier, 3)} vs ${num(control.brier, 3)}`,
      ],
    ];
    const passed = checks.every(([ok]) => ok);
    console.log(`  ${s.variant} vs ${CONTROL}   MAE ${num(v.maeChallenger)} vs ${num(v.maeControl)} over n=${v.n}`);
    for (const [ok, label, detail] of checks) console.log(`      ${ok ? "✓" : "✗"} ${pad(label, 44)} ${detail}`);
    console.log(`      → ${passed ? `${s.variant.toUpperCase()} IS BETTER` : "no detectable difference"}\n`);
  }

  console.log(`  ${c.interpretation.note}\n`);
  if (seasons && seasons.autumn > seasons.summer) {
    console.log(`  ⚠  Most graded nights fall after mid-September, where the held-out test`);
    console.log(`     says the ML advantage does not exist. Report the split with any reading.\n`);
  }
}

// Buckets the finalized target dates by season. This only sorts dates — it does
// not re-grade anything — so it cannot drift from the service's definition of
// an error the way a client-side MAE would.
async function seasonSplit(base: string, days: number): Promise<{ summer: number; autumn: number } | null> {
  try {
    const res = await fetch(`${base}/api/v1/stations/345/predictions?days=${days}&all=1`);
    if (!res.ok) return null;
    const body = (await res.json()) as { predictions: { targetDate: string; variant: string; finalizedAt: string | null }[] };
    const dates = new Set(body.predictions.filter((p) => p.finalizedAt).map((p) => p.targetDate));
    let summer = 0;
    let autumn = 0;
    for (const d of dates) (d.slice(4) >= AUTUMN_FROM ? autumn++ : summer++);
    return { summer, autumn };
  } catch {
    return null; // a presentation nicety; never worth failing the scoreboard over
  }
}

main().catch((e) => {
  console.error(`scoreboard failed: ${e instanceof Error ? e.message : String(e)}`);
  process.exit(1);
});
