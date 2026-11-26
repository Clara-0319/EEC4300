"""What the LLM actually contributes, from the provenance fields the arms record.

An LLM arm can look good for the wrong reason: repair can rewrite its output into valid
graphs, and a second round can hide a bad first answer.  This script separates those:

  repair   how much of the slate had to be edited, and how much editing was needed
  rounds   how many samples needed the reflexive second call, and what it added
  yield    `fill` = emitted/k, i.e. how many slots the LLM actually filled itself
  cost     tokens and dollars per sample and per recovered graph

    python causal/bench/analyze_llm.py --dataset n6_hard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "bench"


def load(arm, k, dataset):
    f = RESULTS_DIR / f"{arm}_{dataset}_k{k}.json"
    return json.load(open(f, encoding="utf-8")) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="n6_hard")
    ap.add_argument("--arms", nargs="*", default=["llm_baseline", "c3gen"])
    args = ap.parse_args()

    for arm in args.arms:
        for k in (5, 20, 50):
            d = load(arm, k, args.dataset)
            if not d:
                continue
            rows = [r for r in d["per_sample"] if "error" not in r]
            n = len(rows)
            agg = d["aggregate"]
            print(f"\n=== {arm}  k={k}  {args.dataset}  n={n} ===")
            print(f"  V={agg['V']:.3f} U={agg['U']:.3f} R={agg['R']:.3f} "
                  f"R/cap={agg['R_norm']:.3f} fill={agg.get('fill',0):.2f} "
                  f"purity={agg.get('purity',0):.3f}")
            print(f"  calls={agg['llm_calls']} tokens={agg['tokens']} "
                  f"${agg['cost_usd']:.4f}  tokens/sample={agg['tokens']/max(n,1):.0f} "
                  f"$ / recovered graph={agg['cost_usd']/max(1e-9, sum(r['n_recovered'] for r in rows)):.5f}")
            if arm == "c3gen":
                rep = [r["info"].get("repair") or {} for r in rows]
                need = [r for r in rep if (r.get("removed", 0) or r.get("added", 0))]
                unrep = sum(r.get("unrepairable", 0) for r in rep)
                print(f"  repair: {len(need)}/{n} samples edited, "
                      f"edges removed={sum(r.get('removed',0) for r in rep)} "
                      f"added={sum(r.get('added',0) for r in rep)} unrepairable={unrep}")
                multi = [r for r in rows if len((r["info"].get("rounds") or [])) > 1]
                gain = 0
                for r in multi:
                    rr = r["info"]["rounds"]
                    gain += rr[-1]["slate"] - rr[0]["slate"]
                print(f"  round 2 used by {len(multi)}/{n} samples, added {gain} graphs total")
                full = sum(1 for r in rows if r.get("fill", 0) >= 0.999)
                print(f"  LLM filled every slot unaided: {full}/{n}")
            else:
                bad = sum(1 for r in rows if r.get("purity", 1) < 0.999)
                print(f"  samples emitting >=1 invalid graph: {bad}/{n} "
                      f"(mean purity {agg.get('purity',0):.3f})")


if __name__ == "__main__":
    main()
