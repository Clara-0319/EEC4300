"""Run an LLM arm on the 3D domain, sequentially, with a hard CNY budget cap.

    python run_llm.py --arm paper --limit 20 --budget-cny 1.2 --tag llm20
    python run_llm.py --arm schema --limit 20 --budget-cny 1.2 --tag llm20

Samples are taken in a fixed stratified order (4 with 1 occupied cell, 8 with 2,
8 with 3) so the 20-sample subset is reproducible without a manifest file.

Money safety:
- every finished sample is checkpointed to the result file immediately, so an
  aborted run never loses the rows it already paid for;
- `--resume PATH` skips ids already present in that file and appends;
- the budget cap is checked before each sample and again after each call.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import core
import llm_arms

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "bench"
DATASET = HERE.parent / "datasets" / "node3x3_h3_all129.json"
CONFIG = HERE.parent.parent / "causal" / "config" / "config_gpt4o.yaml"


def resolve_api_key(explicit: str = "") -> str:
    import os
    if explicit:
        return explicit
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return (cfg.get("llm", {}).get("api_keys", {}) or {}).get("deepseek", "")


def pick_samples(samples, limit: int, seed: int):
    """Stratified: as even as possible across n_blocks, deterministic given seed."""
    rng = random.Random(seed)
    by = {}
    for s in samples:
        by.setdefault(s["n_blocks"], []).append(s)
    for v in by.values():
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < limit:
        added = False
        for b in (1, 2, 3):
            if len(picked) >= limit:
                break
            if i < len(by.get(b, [])):
                picked.append(by[b][i])
                added = True
        if not added:
            break
        i += 1
    return picked


def aggregate(rows, budget=None, extra=None):
    agg = {k: mean(r[k] for r in rows) for k in ("V", "U", "R", "cap", "R_cap", "fill", "purity")}
    agg["n_samples"] = len(rows)
    if budget is not None:
        agg.update({"calls": budget.calls, "prompt_tokens": budget.prompt_tokens,
                    "completion_tokens": budget.completion_tokens,
                    "cost_usd": budget.spent_usd, "cost_cny": budget.spent_cny})
    if extra:
        agg.update(extra)
    by = {}
    for b in (1, 2, 3):
        sub = [r for r in rows if r["n_blocks"] == b]
        if sub:
            by[str(b)] = {"n": len(sub), "V": mean(r["V"] for r in sub),
                          "U": mean(r["U"] for r in sub), "R": mean(r["R"] for r in sub),
                          "R_cap": mean(r["R_cap"] for r in sub)}
    return agg, by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["paper", "schema", "schema_dedupe"])
    ap.add_argument("--slots", default="gt", help="'gt' (=|GT|, the paper default) or an integer")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--budget-cny", type=float, default=1.2, help="hard cap for THIS run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--api-key", default="")
    ap.add_argument("--resume", default="", help="existing result file to append to")
    ap.add_argument("--out", default="", help="result path (default derived from arm/tag/slots)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--thinking", choices=["enabled", "disabled"], default=None,
                    help="DeepSeek thinking mode; default = server default (enabled)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else RESULTS / (
        f"{args.arm}_all129{('_' + args.tag) if args.tag else ''}_k{args.slots}.json")

    rows, cost_used = [], 0.0
    if args.resume:
        prev = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        rows = prev["per_sample"]
        cost_used = prev.get("aggregate", {}).get("cost_cny", 0.0)
        print(f"resume: {len(rows)} rows already paid for ({cost_used:.3f} CNY), appending")

    samples = [s for s in pick_samples(core.load_dataset(args.dataset), args.limit, args.seed)
               if s["observation_set_id"] not in {r["observation_set_id"] for r in rows}]

    def planned(ss):
        return sum(len(core.ground_truth(core.sample_cells(s))) if args.slots == "gt" else int(args.slots)
                   for s in ss)

    print(f"arm={args.arm} todo={len(samples)} planned_calls={planned(samples)} "
          f"cap_this_run={args.budget_cny} CNY")
    if samples:
        print("ids:", ",".join(s["observation_set_id"] for s in samples))
    if args.dry_run:
        for s in samples:
            print(f"  {s['observation_set_id']} blocks={s['n_blocks']} |GT|={s['n_compatible_structures']}")
        return
    if not samples:
        print("nothing to do")
        return

    key = resolve_api_key(args.api_key)
    if not key:
        raise SystemExit("no API key: set DEEPSEEK_API_KEY or llm.api_keys.deepseek in causal/config/config_gpt4o.yaml")
    client = llm_arms.make_client(key)
    budget = llm_arms.Budget(args.budget_cny)

    def flush(stopped: str = ""):
        agg, by = aggregate(rows, budget, {"cost_cny_total": cost_used + budget.spent_cny,
                                           "stopped": stopped})
        payload = {"config": {"arm": args.arm, "slots": args.slots, "n_samples": len(rows),
                              "seed": args.seed, "tag": args.tag,
                              "ids": [r["observation_set_id"] for r in rows],
                              "model": "deepseek-flash", "temperature": 0.7,
                              "thinking": args.thinking},
                   "aggregate": agg, "by_bucket": by, "per_sample": rows}
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)

    for i, s in enumerate(samples, 1):
        if budget.spent_cny > args.budget_cny:
            print("stop: run cap reached before next sample")
            break
        cells = core.sample_cells(s)
        k = len(core.ground_truth(cells)) if args.slots == "gt" else int(args.slots)
        try:
            row = llm_arms.run_arm(args.arm, s, k, client, budget, rng_seed=args.seed,
                                       thinking=args.thinking)
        except RuntimeError as e:
            print(f"stop: {e}")
            break
        rows.append(row)
        flush()
        print(f"  [{len(rows)}] {s['observation_set_id']} k={k} V={row['V']:.2f} U={row['U']:.2f} "
              f"R={row['R']:.3f} R/cap={row['R_cap']:.3f} spent={budget.spent_cny:.3f} CNY", flush=True)

    flush()
    agg, _ = aggregate(rows, budget)
    print(f"\n{args.arm}: n={agg['n_samples']} V={agg['V']:.3f} U={agg['U']:.3f} R={agg['R']:.3f} "
          f"R/cap={agg['R_cap']:.3f} cost_this_run={budget.spent_cny:.3f} CNY")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
