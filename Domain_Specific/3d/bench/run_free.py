"""Run the two zero-cost arms on the 3D domain.

    python run_free.py --arm iid   --budget gt   # independent uniform sampling, paper budget
    python run_free.py --arm slate --budget gt   # G-Slate: enumerate + max-min ordering
    python run_free.py --arm both  --budget 5
    python run_free.py --reuse-curve             # duplicate-probability -> coverage curve

`--budget gt` sets the number of slots to |GT|, which is what the paper's default
--query-multiplier 1.0 does.  `--budget 5` is the equal-budget comparison used in
the causal table (k=5).

Both arms are deterministic given --seed except for `iid`, whose only randomness
is the uniform draw itself; results carry the seed so they can be reproduced.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import core

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "bench"
DATASET = HERE.parent / "datasets" / "node3x3_h3_all129.json"


def run_arm(arm: str, budget: str, samples: List[Dict], seed: int) -> Dict:
    rng = random.Random(seed)
    rows = []
    for i, s in enumerate(samples):
        cells = core.sample_cells(s)
        gt = core.ground_truth(cells)
        k = len(gt) if budget == "gt" else int(budget)
        if arm == "iid":
            emitted = [gt[rng.randrange(len(gt))] for _ in range(k)]
        elif arm == "slate":
            if k >= len(gt):
                picks = list(gt)
                emitted = picks + [picks[j % len(picks)] for j in range(k - len(picks))]
            else:
                emitted = core.greedy_slate(gt, k, seed=seed * 100_000 + i)
        else:
            raise ValueError(arm)
        row = core.score(emitted, cells, gt, k)
        row.update({"observation_set_id": s["observation_set_id"], "n_blocks": s["n_blocks"], "k": k})
        if arm == "slate" and k < len(gt):
            row["min_cheb"] = core.min_pair_distance(emitted)
        rows.append(row)

    agg = {
        key: mean(r[key] for r in rows)
        for key in ("V", "U", "R", "cap", "R_cap", "fill", "purity")
    }
    agg["n_samples"] = len(rows)
    agg["k"] = rows[0]["k"] if len({r["k"] for r in rows}) == 1 else "mixed"
    if arm == "slate":
        chebs = [r["min_cheb"] for r in rows if "min_cheb" in r]
        if chebs:
            agg["mean_min_cheb"] = mean(chebs)

    by_bucket: Dict[str, Dict] = {}
    for b in (1, 2, 3):
        sub = [r for r in rows if r["n_blocks"] == b]
        if sub:
            by_bucket[str(b)] = {
                "n": len(sub),
                "V": mean(r["V"] for r in sub),
                "U": mean(r["U"] for r in sub),
                "R": mean(r["R"] for r in sub),
                "R_cap": mean(r["R_cap"] for r in sub),
            }
    return {"config": {"arm": arm, "budget": budget, "seed": seed}, "aggregate": agg,
            "by_bucket": by_bucket, "per_sample": rows}


def reuse_curve(samples: List[Dict], seed: int = 0, trials: int = 400) -> Dict:
    """Coverage as a function of how often the sampler repeats a previous answer."""
    rng = random.Random(seed)
    out = {}
    for d in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        tot = 0
        for s in samples:
            gt = core.ground_truth(core.sample_cells(s))
            m = len(gt)
            for _ in range(trials):
                seen, keys = [], set()
                for _ in range(m):
                    x = (seen[rng.randrange(len(seen))] if seen and rng.random() < d
                         else gt[rng.randrange(m)])
                    if x not in keys:
                        keys.add(x)
                        seen.append(x)
                tot += len(keys) / m
        out[f"{d:.1f}"] = tot / (trials * len(samples))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["iid", "slate", "both"], default="both")
    ap.add_argument("--budget", default="gt", help="'gt' for |GT| slots, or an integer")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-curve", action="store_true")
    args = ap.parse_args()

    samples = core.load_dataset(args.dataset)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.reuse_curve:
        curve = reuse_curve(samples, seed=args.seed)
        out = RESULTS / "reuse_curve.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(curve, f, indent=1)
        print("reuse_prob -> mean coverage")
        for k, v in curve.items():
            print(f"  d={k}  {v:.4f}")
        print(f"wrote {out}")
        return

    arms = ["iid", "slate"] if args.arm == "both" else [args.arm]
    suffix = "gt" if args.budget == "gt" else str(args.budget)
    for arm in arms:
        res = run_arm(arm, args.budget, samples, args.seed)
        out = RESULTS / f"{arm}_all129_k{suffix}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        a = res["aggregate"]
        print(f"{arm:5s} k={a['k']:>3}  V={a['V']:.3f} U={a['U']:.3f} R={a['R']:.3f} "
              f"cap={a['cap']:.3f} R/cap={a['R_cap']:.3f} fill={a['fill']:.2f}"
              + (f" min_cheb={a['mean_min_cheb']:.2f}" if "mean_min_cheb" in a else "")
              + f"  -> {out.name}")
        for b, row in res["by_bucket"].items():
            print(f"        blocks={b} n={row['n']:>2}  R={row['R']:.3f} R/cap={row['R_cap']:.3f}")


if __name__ == "__main__":
    main()
