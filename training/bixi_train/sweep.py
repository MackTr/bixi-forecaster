"""A capacity check, not a hyperparameter search.

The first training run hit its 600-round cap on every single fit -- early stopping
never fired -- which means the models were still improving when they were cut off.
That matters twice over. The held-out scores understate what the model can do, and
the learning-curve comparison is confounded: "more data" and "more capacity" were
varied together, so a small gain from season two might just be a model that ran
out of room.

This re-runs one fold with the cap lifted and lets early stopping choose. The
answer feeds two decisions: the final round count, and how large the artifact is
allowed to get -- every tree is JSON a Worker parses on a cold start, so accuracy
that costs 4x the bytes for 1% deviance is not free.
"""

from __future__ import annotations

import sys
import time

from . import features, train


def main() -> int:
    df = features.build_matrix()
    fold = train.FOLDS["summer"]
    tr, te = train.split(df, fold)
    print(f"[sweep] summer fold: train={len(tr):,} test={len(te):,}\n")

    configs = [
        ("lr=0.05 cap=600  (as shipped)", {"learning_rate": 0.05}, 600),
        ("lr=0.05 cap=2000", {"learning_rate": 0.05}, 2000),
        ("lr=0.10 cap=2000", {"learning_rate": 0.10}, 2000),
        ("lr=0.10 cap=2000 leaves=127", {"learning_rate": 0.10, "num_leaves": 127}, 2000),
    ]
    for label, params, rounds in configs:
        t0 = time.time()
        b = train.fit(tr, "dep", params=params, rounds=rounds)
        m = train.evaluate(b, tr, te, "dep")
        nodes = sum(t["num_leaves"] * 2 - 1 for t in b.dump_model()["tree_info"])
        mae345, days = train.morning_total_mae(te, train.predict_counts(b, te), "Regina / de Verdun")
        print(
            f"  {label:30s} trees={b.num_trees():>4}  deviance={m['deviance']:.4f}  "
            f"mae={m['mae']:.3f}  345-morning-MAE={mae345:.2f}  ~{nodes:,} nodes  ({time.time() - t0:.0f}s)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
