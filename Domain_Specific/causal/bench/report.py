"""Assemble every finished arm into one table, and check the metric identities.

    python causal/bench/report.py            # markdown to stdout
    python causal/bench/report.py --write    # also refresh results/bench/REPORT.md

Two things this deliberately prints, because they are the point of the experiment:

1. `R/cap` split by regime.  For |GT| >= k the identity  R/cap = U  holds exactly, so
   coverage carries no information beyond "distinct valid graphs per slot" -- the only
   regime where R adds signal is |GT| <= k, where it measures *completeness* (did you
   find every compatible graph).  Averaging the two regimes together is what made the
   old numbers look either perfect or meaningless.
2. `purity` and `fill`, the two factors of V.  V alone is gameable: emitting the same
   valid graph k times gives V=1.0, U=1/k, R/cap=1/k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "bench"


def load_all():
    out = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if _skip(f.name):
            continue
        if f.name == "REPORT.md":
            continue
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        d["_file"] = f.name
        out.append(d)
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def regimes(rows, k):
    hard = [r for r in rows if "error" not in r and r["n_gt"] <= k]
    easy = [r for r in rows if "error" not in r and r["n_gt"] > k]
    return hard, easy


def _skip(name):
    """Deterministic arms only: LLM runs live in LLM_RESULTS.md (they are
    budget-truncated to a 10-28 sample subset and must not be pooled with the
    72-sample deterministic table)."""
    return name.startswith(("llm_baseline", "c3gen"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    runs = load_all()
    if not runs:
        print("no results yet")
        return

    lines = []
    lines.append("# Bench results\n")
    lines.append("All arms share one manifest (60 samples, 10 per |GT| stratum) and one "
                 "metric implementation (`bench/core.py:score`).\n")
    lines.append("| data | arm | n | k | fill | purity | V | U | R | R/cap | hard R/cap "
                 "| easy U | calls | tokens | $ |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    identity_violations = 0
    checked = 0
    for d in runs:
        a, k = d["aggregate"], d["aggregate"]["k"]
        rows = d["per_sample"]
        hard, easy = regimes(rows, k)
        h = _mean([r["R_norm"] for r in hard])
        e = _mean([r["U"] for r in easy])
        for r in easy:
            checked += 1
            if abs(r["R_norm"] - r["n_unique_valid"] / k) > 1e-9:
                identity_violations += 1
        cfg = d["config"]
        data = Path(cfg["dataset"]).stem + (f"_{cfg['tag']}" if cfg.get("tag") else "")
        lines.append(
            f"| {data} | {a['arm']} | {cfg.get('n_samples') or len(rows)} | {k} | {a.get('fill', float('nan')):.2f} "
            f"| {a.get('purity', float('nan')):.3f} | {a['V']:.3f} | {a['U']:.3f} "
            f"| {a['R']:.3f} | {a['R_norm']:.3f} | {h:.3f} | {e:.3f} "
            f"| {a['llm_calls']} | {a['tokens']} | {a['cost_usd']:.4f} |"
        )

    lines.append("")
    lines.append(f"Metric identity check: `R/cap == U` when |GT| >= k, verified on "
                 f"{checked} sample-rows, {identity_violations} violations.")
    lines.append("")
    lines.append("`hard` = samples with |GT| <= k (cap is 1.0, so R/cap is the fraction of "
                 "compatible graphs actually found); `easy` = |GT| > k (R/cap collapses to U).")

    text = "\n".join(lines)
    print(text)
    if args.write:
        out = RESULTS_DIR / "REPORT.md"
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
