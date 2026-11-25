"""Build an observation-set dataset at larger n, with exact ground truth.

The published 5-node set was produced by enumerating 29 281 DAGs per sample and keeping
the compatible ones.  That is 3 781 503 per sample at n=6 and grows as the number of
labelled DAGs, so it does not extend.  With the admissibility theorem we instead search
only the whitelist, which makes |GT| affordable -- and we deliberately keep only sets
whose |GT| is small enough to enumerate exhaustively, since those are the sets where
completeness is a real search problem (the |GT| <= k regime).

    python causal/bench/make_dataset.py --n 6 --target 120 --max-gt 200 \
        --out causal/datasets/node06/n6_hard.json

The ground truth stored here is exact (full enumeration of the whitelisted space), not
sampled, so it can be used for scoring without qualification.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import core  # noqa: E402


def random_dag(nodes, rng, p):
    order = list(nodes)
    rng.shuffle(order)
    return frozenset((order[i], order[j])
                     for i in range(len(order)) for j in range(i + 1, len(order))
                     if rng.random() < p)


def observations_from(nodes, g, rng, n_obs):
    adj = core.adjacency(nodes, g)
    obs = []
    for p in sorted(rng.sample(list(nodes), n_obs)):
        eff = {x: (1 if x in core.reach_from(adj, p) else 0) for x in nodes}
        obs.append({
            "perturbed_node": p,
            "effects": eff,
            "string": f"Perturb({p}) -> " + " ".join(f"{x}:{v}" for x, v in eff.items()),
        })
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--max-gt", type=int, default=200,
                    help="keep only sets with |GT| <= this, so ground truth is exhaustive")
    ap.add_argument("--n-obs", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--max-edges", type=int, default=None)
    ap.add_argument("--edge-prob", type=float, default=0.35)
    ap.add_argument("--attempts", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=33550336)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    nodes = [chr(ord("A") + i) for i in range(args.n)]
    max_edges = args.max_edges if args.max_edges is not None else args.n + 6
    rng = random.Random(args.seed)
    seen = {}
    attempts = 0
    while len(seen) < args.target and attempts < args.attempts:
        attempts += 1
        n_obs = rng.choice(args.n_obs)
        g = random_dag(nodes, rng, args.edge_prob)
        obs = observations_from(nodes, g, rng, n_obs)
        sig = "|".join(o["string"] for o in obs)
        if sig in seen:
            continue
        c = core.Constraints.from_observations(nodes, obs)
        # one extra solution is enough to know the set is too large to keep
        sols = core.enumerate_all(c, limit=args.max_gt + 1)
        n_gt = len(sols)
        if n_gt > args.max_gt:
            continue
        seen[sig] = {
            "observation_set_id": None,  # assigned below, stable order needed
            "n_observations": len(obs),
            "observations": obs,
            "ground_truth_graphs": [
                {"nodes": nodes, "edges": sorted([u, v] for (u, v) in s)} for s in sols
            ],
            "n_compatible_graphs": n_gt,
            "generating_graph": sorted([u, v] for (u, v) in g),
        }
        if attempts % 200 == 0:
            print(f"  {len(seen)}/{args.target} kept after {attempts} attempts", flush=True)

    records = sorted(seen.values(), key=lambda r: (r["n_observations"], r["n_compatible_graphs"]))
    by_obs = {}
    for i, r in enumerate(records, start=1):
        r["observation_set_id"] = f"n{r['n_observations']}_{i:03d}"
        r["nodes"] = nodes
        r["max_edges"] = max_edges
        by_obs.setdefault(str(r["n_observations"]), []).append(r)

    out = {
        "metadata": {
            "nodes": nodes,
            "max_edges": max_edges,
            "hypothesis_space_size": None,
            "total_observation_sets": len(records),
            "generation_method": "whitelist_exhaustive_gt",
            "complete_gt_coverage": True,
            "gt_cap": args.max_gt,
            "seed": args.seed,
            "edge_prob": args.edge_prob,
            "note": ("every ground truth is the complete set of compatible DAGs "
                     "(enumeration over the admissible-edge whitelist); sets with "
                     f"|GT| > {args.max_gt} were rejected rather than truncated"),
        },
        "datasets_by_n_observations": by_obs,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    gts = sorted(r["n_compatible_graphs"] for r in records)
    print(f"wrote {path}  n={args.n}  sets={len(records)}  attempts={attempts}  "
          f"|GT| min={gts[0]} median={gts[len(gts)//2]} max={gts[-1]}")
    print(f"|GT| <= 5: {sum(g <= 5 for g in gts)}   <= 20: {sum(g <= 20 for g in gts)}   "
          f"<= 50: {sum(g <= 50 for g in gts)}")
    return records


if __name__ == "__main__":
    main()
