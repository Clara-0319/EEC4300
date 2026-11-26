"""Re-score an existing result file on a subset of samples (free, no API calls).

The paid arms run on different numbers of samples (budget), but a paired comparison needs
the *same* samples.  Since every manifest is nested (tiny12 < sub36 < n6), restricting a
broader run to the narrow sample list gives the paired numbers without paying twice.

    python causal/bench/restrict.py --file c3gen_n6_hard_sub36_k20.json \
        --ids causal/datasets/node06/manifest_n6_tiny12.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "bench"


def aggregate(rows):
    ok = [r for r in rows if "error" not in r]
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0
    out = {k: mean([r.get(k, 0) for r in ok])
           for k in ("V", "U", "R", "cap", "R_norm", "purity", "fill", "slot_fill")}
    out["n"] = len(ok)
    out["tokens"] = sum(r.get("tokens", 0) for r in ok)
    out["cost"] = sum(r.get("cost", 0.0) for r in ok)
    out["calls"] = sum(r.get("n_calls", 0) for r in ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--ids", required=True, help="manifest json whose sample ids define the subset")
    args = ap.parse_args()

    d = json.load(open(RESULTS_DIR / args.file, encoding="utf-8"))
    keep = {s["id"] for s in json.load(open(args.ids, encoding="utf-8"))["samples"]}
    rows = [r for r in d["per_sample"] if r["id"] in keep]
    a = aggregate(rows)
    print(f"{d['aggregate']['arm']:<14} k={d['aggregate']['k']:<3} n={a['n']:<3} "
          f"V={a['V']:.3f} purity={a['purity']:.3f} fill={a['fill']:.2f} "
          f"U={a['U']:.3f} R={a['R']:.3f} R/cap={a['R_norm']:.3f} ${a['cost']:.4f}")
    widths = {b: [r for r in rows if r["bucket"] == b] for b in {r["bucket"] for r in rows}}
    for b, v in sorted(widths.items(), key=lambda kv: int(kv[0].split("-")[0])):
        print(f"   |GT| {b:>8s} n={len(v):2d} V={sum(x['V'] for x in v)/len(v):.3f} "
              f"R/cap={sum(x['R_norm'] for x in v)/len(v):.3f}")


if __name__ == "__main__":
    main()
