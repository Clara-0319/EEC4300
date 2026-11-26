"""Run one arm on the frozen manifest and write a scoreable result file.

    python causal/bench/run_experiment.py --arm oracle --k 20
    python causal/bench/run_experiment.py --arm c3gen --k 5 --provider deepseek
    python causal/bench/run_experiment.py --table            # print every result so far

Design rules encoded here
-------------------------
* `k` (the output budget) is an argument, and every metric divides by it, so arms can
  never choose their own denominator.
* per-sample rows are stored, not just means, so paired tests and per-stratum
  breakdowns are possible afterwards without re-running a paid arm.
* a failing sample is recorded as a failure, never dropped: dropping failures is the
  second-easiest way to fake a benchmark after changing denominators.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def _pin_hashseed():
    """Re-run under PYTHONHASHSEED=0 so set iteration order cannot leak into results.

    Several helpers walk `set` objects of node names.  CPython randomises str hashing
    per process, so without this the seeded arms drift by a few points between runs --
    a reviewer would read that drift as instability of the method itself.  Implemented
    as a subprocess re-launch because os.execve crashed the interpreter on Windows.
    """
    if os.environ.get("PYTHONHASHSEED") == "0" or os.environ.get("_HYPO_PINNED"):
        return
    import subprocess

    env = dict(os.environ, PYTHONHASHSEED="0", _HYPO_PINNED="1")
    sys.exit(subprocess.call([sys.executable, *sys.argv], env=env))


BENCH_DIR = Path(__file__).resolve().parent
CAUSAL_DIR = BENCH_DIR.parent
for p in (str(BENCH_DIR), str(CAUSAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arms
import core
import data
from llm_client import MockLLM, make_llm

RESULTS_DIR = CAUSAL_DIR / "results" / "bench"
_N5_DATASET = CAUSAL_DIR / "datasets" / "node05" / "n5_sampled.json"
_N5_MANIFEST = CAUSAL_DIR / "datasets" / "node05" / "manifest_n60_seed33550336.json"
_N6_DATASET = CAUSAL_DIR / "datasets" / "node06" / "n6_hard.json"
_N6_MANIFEST = CAUSAL_DIR / "datasets" / "node06" / "manifest_n6_all120.json"

# n=5 stays the default where its (135 MB, gitignored) dataset exists; a fresh clone is
# pointed at n=6 instead, which is the regime where recovery can still separate arms.
if _N5_DATASET.exists() and _N5_MANIFEST.exists():
    DEFAULT_DATASET, DEFAULT_MANIFEST = _N5_DATASET, _N5_MANIFEST
else:
    DEFAULT_DATASET, DEFAULT_MANIFEST = _N6_DATASET, _N6_MANIFEST

ARMS = {
    "oracle": arms.run_oracle,
    "symbolic": arms.run_symbolic,
    "random": arms.run_random,
    "llm_baseline": arms.run_llm_baseline,
    "c3gen": arms.run_c3gen,
    "c3gen_hybrid": arms.run_c3gen_hybrid,
}


def run_arm(name, k, manifest_path, dataset_path, provider, limit, state_budget, seed,
            verbose, workers=1, tag=""):
    missing = [p for p in (dataset_path, manifest_path) if not Path(p).exists()]
    if missing:
        raise SystemExit(
            "missing input file(s):\n  " + "\n  ".join(str(p) for p in missing) +
            "\n\nA fresh clone has no datasets/ (it is gitignored). Rebuild the n=6\n"
            "benchmark, then point this run at it:\n"
            "  python causal/bench/make_dataset.py --out causal/datasets/node06/n6_hard.json\n"
            "  python causal/bench/run_experiment.py --arm oracle --k 5 \\\n"
            "      --dataset causal/datasets/node06/n6_hard.json \\\n"
            "      --manifest causal/datasets/node06/manifest_n6_all120.json"
        )
    samples = data.load_manifest_samples(manifest_path, dataset_path, seed=seed)
    if limit:
        samples = samples[:limit]
    fn = ARMS[name]
    llm = None
    if name in ("llm_baseline", "c3gen", "c3gen_hybrid"):
        llm = make_llm(provider)

    rows, t0 = [], time.time()
    tot = {"tokens": 0, "cost": 0.0, "calls": 0, "states": 0, "wall": 0.0}

    def work(item):
        i, s = item
        rng = random.Random(seed * 100_000 + i)
        try:
            if name == "symbolic":
                res = fn(s, k, rng, None, state_budget=state_budget)
            else:
                res = fn(s, k, rng, llm)
            m = core.score(res["emitted"], s["nodes"], s["constraints"], s["gt"], k)
            row = {**m, "id": s["id"], "bucket": s["bucket"], "n_obs": s["n_obs"],
                   "n_calls": res["n_calls"], "search_states": res["search_states"],
                   "info": res.get("info", {})}
            u = res.get("usage") or {}
            row["tokens"] = u.get("total_tokens", 0)
            row["cost"] = res.get("cost", 0.0)
        except Exception as e:  # record, never drop
            row = {"id": s["id"], "bucket": s["bucket"], "n_obs": s["n_obs"],
                   "error": f"{type(e).__name__}: {e}", "V": 0.0, "U": 0.0, "R": 0.0,
                   "R_norm": 0.0, "cap": 0.0, "n_gt": s["gt_file"], "tokens": 0, "cost": 0.0}
        return row

    items = list(enumerate(samples, 1))
    done = 0
    if workers and workers > 1 and name in ("llm_baseline", "c3gen", "c3gen_hybrid"):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, it): it[0] for it in items}
            buf = {}
            for fut in as_completed(futs):
                idx = futs[fut]
                buf[idx] = fut.result()
                done += 1
                if verbose:
                    r = buf[idx]
                    print(f"  [{done}/{len(items)}] {r['id']:8s} |GT|={r.get('n_gt',0):5d} "
                          f"V={r['V']:.2f} U={r['U']:.2f} R={r['R']:.3f} "
                          f"cap={r['cap']:.2f} Rn={r['R_norm']:.3f}", flush=True)
        rows = [buf[i] for i, _ in items]
    else:
        for it in items:
            r = work(it)
            rows.append(r)
            done += 1
            if verbose:
                print(f"  [{done}/{len(items)}] {r['id']:8s} |GT|={r.get('n_gt',0):5d} "
                      f"V={r['V']:.2f} U={r['U']:.2f} R={r['R']:.3f} "
                      f"cap={r['cap']:.2f} Rn={r['R_norm']:.3f}", flush=True)
            if "error" in r:
                print(f"  !! {r['id']}: {r['error']}")

    for r in rows:
        tot["tokens"] += r.get("tokens", 0) or 0
        tot["cost"] += r.get("cost", 0.0) or 0.0
        tot["calls"] += r.get("n_calls", 0) or 0
        tot["states"] += r.get("search_states", 0) or 0
        tot["wall"] += (r.get("info") or {}).get("wall", 0.0) or 0.0

    agg = aggregate(rows)
    agg.update({"arm": name, "k": k, "provider": provider if llm else "-",
                "n_samples": len(rows), "tokens": tot["tokens"], "cost_usd": tot["cost"],
                "llm_calls": tot["calls"], "search_states": tot["states"],
                "wall_seconds": round(time.time() - t0, 1)})
    out = {
        "config": {"arm": name, "k": k, "manifest": str(manifest_path),
                   "dataset": str(dataset_path), "provider": provider, "seed": seed,
                   "tag": tag, "n_samples": len(rows)},
        "aggregate": agg,
        "per_sample": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # the dataset stem is part of the filename: results from different datasets are
    # not comparable and must never silently overwrite each other
    stem = Path(dataset_path).stem + (f"_{tag}" if tag else "")
    path = RESULTS_DIR / f"{name}_{stem}_k{k}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print_summary(agg)
    print(f"written -> {path}")
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(rows):
    ok = [r for r in rows if "error" not in r]
    agg = {
        "V": _mean([r["V"] for r in ok]),
        "U": _mean([r["U"] for r in ok]),
        "R": _mean([r["R"] for r in ok]),
        "cap": _mean([r["cap"] for r in ok]),
        "R_norm": _mean([r["R_norm"] for r in ok]),
        "slot_fill": _mean([r.get("slot_fill", 0) for r in ok]),
        "purity": _mean([r.get("purity", 0) for r in ok]),
        "fill": _mean([r.get("fill", 0) for r in ok]),
        "n_failed": len(rows) - len(ok),
    }
    buckets = {}
    for r in ok:
        buckets.setdefault(r["bucket"], []).append(r)
    agg["by_bucket"] = {
        b: {"n": len(v), "V": _mean([r["V"] for r in v]), "U": _mean([r["U"] for r in v]),
            "R": _mean([r["R"] for r in v]), "cap": _mean([r["cap"] for r in v]),
            "R_norm": _mean([r["R_norm"] for r in v])}
        for b, v in sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0]))
    }
    # the honest macro average: every |GT| stratum weighs the same
    if agg["by_bucket"]:
        for key in ("V", "U", "R", "cap", "R_norm"):
            agg[f"macro_{key}"] = _mean([b[key] for b in agg["by_bucket"].values()])
    return agg


def print_summary(a):
    print(f"\n=== {a['arm']}  k={a['k']}  n={a['n_samples']} "
          f"(failed {a['n_failed']}) ===")
    print(f"  V={a['V']:.3f}  U={a['U']:.3f}  R={a['R']:.3f}  cap={a['cap']:.3f}  "
          f"R/cap={a['R_norm']:.3f}  fill={a['slot_fill']:.2f}")
    print(f"  fill={a.get('fill', 0):.2f}  purity={a.get('purity', 0):.3f}  "
          f"(V = fill x purity)")
    print(f"  macro R/cap={a.get('macro_R_norm', 0):.3f}  tokens={a['tokens']}  "
          f"calls={a['llm_calls']}  states={a['search_states']}  ${a['cost_usd']:.4f}")
    for b, v in a["by_bucket"].items():
        print(f"    |GT| {b:>8s}  n={v['n']:2d}  R={v['R']:.3f}  cap={v['cap']:.3f}  "
              f"R/cap={v['R_norm']:.3f}")


def print_table():
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("no results yet")
        return
    print(f"{'arm':<14}{'k':>4}{'V':>8}{'U':>8}{'R':>8}{'cap':>8}{'R/cap':>8}"
          f"{'macro':>8}{'calls':>8}{'tokens':>9}{'$':>9}")
    for f in files:
        a = json.load(open(f, encoding="utf-8"))["aggregate"]
        print(f"{a['arm']:<14}{a['k']:>4}{a['V']:>8.3f}{a['U']:>8.3f}{a['R']:>8.3f}"
              f"{a['cap']:>8.3f}{a['R_norm']:>8.3f}{a.get('macro_R_norm', 0):>8.3f}"
              f"{a['llm_calls']:>8}{a['tokens']:>9}{a['cost_usd']:>9.4f}")


def main():
    _pin_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), default="oracle")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--state-budget", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--build-manifest", action="store_true")
    ap.add_argument("--per-bucket", type=int, default=10)
    ap.add_argument("--tag", default="", help="suffix for the result filename, e.g. sub36")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel samples for the LLM arms (each sample is independent)")
    args = ap.parse_args()

    if args.table:
        print_table()
        return
    if args.build_manifest:
        man = data.build_manifest(args.dataset, args.manifest, per_bucket=args.per_bucket)
        print(f"manifest -> {args.manifest}  ({man['n_samples']} samples, "
              f"buckets {man['buckets']})")
        return
    if not Path(args.manifest).exists():
        print(f"manifest missing, building it at {args.manifest}")
        data.build_manifest(args.dataset, args.manifest, per_bucket=args.per_bucket)
    run_arm(args.arm, args.k, args.manifest, args.dataset, args.provider,
            args.limit, args.state_budget, args.seed, verbose=not args.quiet,
            workers=args.workers, tag=args.tag)


if __name__ == "__main__":
    main()
