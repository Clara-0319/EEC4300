"""Paired statistics for the arm comparison.

Every arm is run on the *same* frozen manifest, so sample `id` pairs rows across arms and
a paired test is the right instrument: unpaired tests would be dominated by the |GT|
spread (1 to 192 here) rather than by the method.

Reported per pair of arms:
  * mean difference in `R_norm` (equals `U` above the cap, = completeness below it),
  * bootstrap 95% CI of that mean difference over samples (10 000 resamples),
  * Wilcoxon signed-rank p-value (two-sided), skipped when every difference is 0,
  * win/tie/loss counts, which are the most legible summary for a small n.

Usage:
    python causal/bench/stats.py --a llm_baseline --b c3gen --k 5 --dataset n6_hard
    python causal/bench/stats.py --all --dataset n6_hard --metric R_norm
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR.parent / "results" / "bench"


def load(arm, k, dataset, tag=""):
    f = RESULTS_DIR / f"{arm}_{dataset}{'_' + tag if tag else ''}_k{k}.json"
    if not f.exists():
        return None
    d = json.load(open(f, encoding="utf-8"))
    rows = {}
    for r in d["per_sample"]:
        rows[r["id"]] = {**r, "dataset": dataset, "arm": d["aggregate"]["arm"],
                         "k": d["aggregate"]["k"]}
    return d["aggregate"], rows


def bootstrap_ci(xs, n_boot=10_000, seed=0, alpha=0.05):
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


def compare(rows_a, rows_b, metric="R_norm"):
    ids = sorted(set(rows_a) & set(rows_b))
    diffs = [rows_b[i][metric] - rows_a[i][metric] for i in ids
             if "error" not in rows_a[i] and "error" not in rows_b[i]]
    if not diffs:
        return None
    wins = sum(1 for d in diffs if d > 1e-9)
    ties = sum(1 for d in diffs if abs(d) <= 1e-9)
    losses = sum(1 for d in diffs if d < -1e-9)
    lo, hi = bootstrap_ci(diffs)
    p = None
    try:
        from scipy.stats import wilcoxon

        if any(abs(d) > 1e-12 for d in diffs):
            p = float(wilcoxon(diffs, zero_method="wilcox").pvalue)
    except Exception:
        pass
    return {"n": len(diffs), "mean_diff": sum(diffs) / len(diffs),
            "ci95": (lo, hi), "p_wilcoxon": p,
            "wins_b": wins, "ties": ties, "losses_b": losses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="n6_hard")
    ap.add_argument("--metric", default="R_norm")
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        arms = sorted({p.name.split("_")[0] for p in RESULTS_DIR.glob(f"*_{args.dataset}_k*.json")})
        for k in (5, 20, 50):
            print(f"\n### {args.dataset}  k={k}  metric={args.metric}")
            print(f"{'arm':<14}{'n':>4}{'mean':>8}{'macro':>8}")
            loaded = {}
            for arm in arms:
                got = load(arm, k, args.dataset)
                if not got:
                    continue
                agg, rows = got
                loaded[arm] = rows
                macro = agg.get(f"macro_{args.metric}", agg.get(args.metric, 0))
                print(f"{arm:<14}{len(rows):>4}{agg.get(args.metric, 0):>8.3f}{macro:>8.3f}")
            names = [a for a in loaded]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    res = compare(loaded[names[i]], loaded[names[j]], args.metric)
                    if not res:
                        continue
                    p = "n/a" if res["p_wilcoxon"] is None else f"{res['p_wilcoxon']:.2e}"
                    print(f"  {names[j]} - {names[i]}: {res['mean_diff']:+.3f} "
                          f"[{res['ci95'][0]:+.3f},{res['ci95'][1]:+.3f}] p={p} "
                          f"W/T/L={res['wins_b']}/{res['ties']}/{res['losses_b']}")
        return

    got_a = load(args.a, args.k, args.dataset)
    got_b = load(args.b, args.k, args.dataset)
    if not got_a or not got_b:
        raise SystemExit("missing result file(s)")
    res = compare(got_a[1], got_b[1], args.metric)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
