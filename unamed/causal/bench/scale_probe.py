"""How expensive is exact coverage when n grows?  Measures time-to-N-solutions.

The n=5 dataset is cheap enough that exhaustive whitelisted search *is* the oracle
(9 339 candidate graphs over 60 samples).  That is also why no LLM-based arm can win at
n=5.  This probe asks where that stops being true, by timing `enumerate_all` under a
solution cap at larger n:

    python -u causal/bench/scale_probe.py --n 6 --n-obs 1 2 3 --limit 1000

A cap is used instead of a wall-clock timeout because the ratio (solutions found) /
(seconds) is what decides whether a dataset of `S` samples is buildable.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import core  # noqa: E402


def random_dag(nodes, rng, p=0.35):
    order = list(nodes)
    rng.shuffle(order)
    return frozenset((order[i], order[j])
                     for i in range(len(order)) for j in range(i + 1, len(order))
                     if rng.random() < p)


def observations_from(nodes, g, rng, n_obs):
    adj = core.adjacency(nodes, g)
    obs = []
    for p in rng.sample(list(nodes), n_obs):
        eff = {x: (1 if x in core.reach_from(adj, p) else 0) for x in nodes}
        obs.append({"perturbed_node": p, "effects": eff,
                    "string": f"Perturb({p}) -> " + " ".join(f"{x}:{v}" for x, v in eff.items())})
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--n-obs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    nodes = [chr(ord("A") + i) for i in range(args.n)]
    rng = random.Random(args.seed)
    for n_obs in args.n_obs:
        for rep in range(args.repeats):
            g = random_dag(nodes, rng)
            c = core.Constraints.from_observations(nodes, observations_from(nodes, g, rng, n_obs))
            t = time.time()
            sols = core.enumerate_all(c, limit=args.limit)
            dt = time.time() - t
            hit_cap = len(sols) >= args.limit
            print(f"n={args.n} n_obs={n_obs} rep={rep} allowed={len(c.allowed)}/{args.n*(args.n-1)} "
                  f"solutions={len(sols)}{'+' if hit_cap else ''} time={dt:.2f}s "
                  f"({len(sols)/max(dt,1e-9):.0f} sol/s)", flush=True)


if __name__ == "__main__":
    main()
