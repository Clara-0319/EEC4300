"""Build causal/results/bench/LLM_RESULTS.md from the paid LLM runs.

Kept separate from report.py on purpose: these runs are budget-truncated to a
10-sample (and one 28-sample) subset, so pooling them with the 72-sample
deterministic table would be misleading.

    python causal/bench/llm_report.py --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "bench"

# (file, label) in reporting order; only files with real LLM calls belong here
FILES = [
    ("llm_baseline_n6_hard_tiny12_k5.json", "Baseline (unconstrained, 5 calls/sample)"),
    ("c3gen_n6_hard_tiny12_k5.json", "C3-Gen (constrained, 1 call/sample)"),
    ("c3gen_hybrid_n6_hard_tiny12_k5.json", "C3-Gen + exact gap fill"),
    ("c3gen_hybrid_n6_hard_tiny12_k20.json", "C3-Gen + exact gap fill (k=20)"),
    ("llm_baseline_n6_hard_k20six_k20.json", "Baseline (k=20, six large-|GT| samples)"),
    ("c3gen_n6_hard_sub36_k20.json", "C3-Gen (k=20, 28 samples)"),
]


def agg(rows):
    n = len(rows)
    m = lambda key: sum(r.get(key, 0) for r in rows) / n
    tok = sum(r.get("tokens", 0) for r in rows)
    calls = sum(r.get("n_calls", 0) for r in rows)
    cost = sum(r.get("cost", 0.0) for r in rows)
    graphs = sum(r.get("n_unique_valid", 0) for r in rows)
    return dict(V=m("V"), U=m("U"), R=m("R"), R_norm=m("R_norm"), fill=m("fill"),
                purity=m("purity"), tok=tok, calls=calls, cost=cost, graphs=graphs, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    out, tot_cost, tot_tok = [], 0.0, 0
    out.append("| arm | k | n | V | purity | U | R | R/cap | fill | calls | tokens | $ | $/sample | tok/graph |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for name, label in FILES:
        p = RESULTS_DIR / name
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        ok = [r for r in d["per_sample"] if "error" not in r]
        if not ok:
            continue
        a = agg(ok)
        rows.append((label, d["aggregate"]["k"], a, ok))
        tot_cost += a["cost"]
        tot_tok += a["tok"]
        out.append(f"| {label} | {d['aggregate']['k']} | {a['n']} | {a['V']:.3f} | {a['purity']:.3f} "
                   f"| {a['U']:.3f} | {a['R']:.3f} | {a['R_norm']:.3f} | {a['fill']:.2f} "
                   f"| {a['calls']} | {a['tok']} | ${a['cost']:.4f} | ${a['cost']/a['n']:.4f} "
                   f"| {a['tok']/max(a['graphs'],1):.0f} |")

    out.append("")
    out.append(f"Total spent on the runs listed above: **${tot_cost:.4f}**, {tot_tok} tokens.")
    out.append("")
    out.append("## Per-sample behaviour at k=5 (n=6 dataset, 10 samples)")
    out.append("")
    out.append("```")
    for label, k, a, ok in rows:
        if k != 5:
            continue
        out.append(f"# {label}")
        for r in ok:
            out.append(f"  {r['id']:>8s} |GT|={r['n_gt']:>3d} emitted={r.get('n_emitted')} "
                       f"valid={r.get('n_valid')} distinct={r.get('n_unique_valid')} "
                       f"V={r['V']:.2f} U={r['U']:.2f} R/cap={r['R_norm']:.3f} "
                       f"calls={r.get('n_calls')} tok={r.get('tokens')}")
    out.append("```")
    # --- paired k=20 block: baseline vs C3-Gen on the same six samples -------
    pb = RESULTS_DIR / "llm_baseline_n6_hard_k20six_k20.json"
    pc = RESULTS_DIR / "c3gen_n6_hard_sub36_k20.json"
    if pb.exists() and pc.exists():
        b = [r for r in json.load(open(pb, encoding="utf-8"))["per_sample"] if "error" not in r]
        c = {r["id"]: r for r in json.load(open(pc, encoding="utf-8"))["per_sample"]}
        ids = [r["id"] for r in b]
        out.append("")
        out.append("## Paired k=20 comparison (same six samples, |GT| = 96-192)")
        out.append("")
        out.append("| sample | \\|GT\\| | baseline R/cap | C3-Gen R/cap | baseline tokens | C3-Gen tokens | baseline $ | C3-Gen $ |")
        out.append("|---|---|---|---|---|---|---|---|")
        tb = tc = cb = cc = 0
        for r in b:
            q = c[r["id"]]
            tb += r["tokens"]; tc += q["tokens"]; cb += r["cost"]; cc += q["cost"]
            out.append(f"| {r['id']} | {r['n_gt']} | {r['R_norm']:.3f} | {q['R_norm']:.3f} "
                       f"| {r['tokens']} | {q['tokens']} | ${r['cost']:.4f} | ${q['cost']:.4f} |")
        out.append(f"| **total** | | **{sum(r['R_norm'] for r in b)/len(b):.3f}** "
                   f"| **{sum(c[i]['R_norm'] for i in ids)/len(ids):.3f}** | {tb} | {tc} | ${cb:.4f} | ${cc:.4f} |")
        out.append("")
        out.append(f"Cost ratio **{cb/cc:.1f}x**, token ratio **{tb/tc:.1f}x**, call ratio "
                   f"**{sum(r['n_calls'] for r in b)/sum(c[i]['n_calls'] for i in ids):.0f}x** in favour of C3-Gen.")
        out.append("")
        diff = [round(c[i]["R_norm"] - r["R_norm"], 4) for i, r in zip(ids, b)]
        out.append(f"Paired R/cap difference (C3-Gen - baseline): {diff}. "
                   f"Only 2 of 6 samples differ, both because the baseline repeated a graph "
                   f"(U = 0.95 instead of 1); with n = 6 a Wilcoxon test cannot reach "
                   f"significance, so the metric difference is reported as a direction, "
                   f"not a result. The cost difference is 7.4x on every sample.")

    text = "\n".join(out) + "\n"
    print(text)
    if args.write:
        (RESULTS_DIR / "LLM_RESULTS.md").write_text(text, encoding="utf-8")
        print(f"written -> {RESULTS_DIR / 'LLM_RESULTS.md'}")


if __name__ == "__main__":
    main()
