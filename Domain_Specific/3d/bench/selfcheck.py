"""Five independent checks for the 3D bench.  All must pass before any number is cited.

1. dataset reproduces: build_dataset() again == the JSON on disk (no drift, no seed).
2. |GT| == 3^k and every stored ground truth is distinct and top-view consistent.
3. the repository's own enumerator (3d/generate_3d_dataset_complete.py) returns
   exactly the same structure set as core.ground_truth, for all 129 sets.
4. G-Slate at the paper budget emits exactly GT: V=U=R=1.000 on every sample.
5. i.i.d. sampling at the paper budget lands on the analytic coupon-collector
   ceiling 1-(1-1/m)^m within Monte-Carlo error (checked per |GT| value).

    python selfcheck.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import core

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "datasets" / "node3x3_h3_all129.json"
REPO = HERE.parent / "generate_3d_dataset_complete.py"

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def repo_structures(top_view: np.ndarray, max_height: int):
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen3d", REPO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [np.stack(s.layers) for s in mod.enumerate_all_structures_by_heights(top_view, max_height)]


def main() -> None:
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    samples = raw["samples"]
    rebuilt = core.build_dataset()["samples"]

    # 1 ---------------------------------------------------------------
    same = len(samples) == len(rebuilt) and all(
        a["observation_set_id"] == b["observation_set_id"]
        and a["ground_truth_structures"] == b["ground_truth_structures"]
        for a, b in zip(samples, rebuilt)
    )
    check("dataset reproduces from code", same, f"{len(samples)} sets")

    # 2 ---------------------------------------------------------------
    bad = []
    for s in samples:
        cells = core.sample_cells(s)
        gt = [tuple(h) for h in s["ground_truth_structures"]]
        if len(gt) != 3 ** len(cells):
            bad.append((s["observation_set_id"], "size", len(gt), 3 ** len(cells)))
        if len(set(gt)) != len(gt):
            bad.append((s["observation_set_id"], "duplicate gt"))
        if s["top_view"] != core.top_view_of(cells):
            bad.append((s["observation_set_id"], "top view mismatch"))
        if set(gt) != set(core.ground_truth(cells)):
            bad.append((s["observation_set_id"], "gt != product set"))
    check("|GT| = 3^k, distinct, top-view consistent", not bad, str(bad[:3]))

    # 3 ---------------------------------------------------------------
    mism = []
    for s in samples:
        cells = core.sample_cells(s)
        tv = np.array([[int(ch) for ch in s["top_view"][r * 3:(r + 1) * 3]] for r in range(3)])
        repo = repo_structures(tv, 3)
        mine = set(core.ground_truth(cells))
        theirs = set()
        for arr in repo:
            heights = []
            for r, c in cells:
                heights.append(int(arr[:, r, c].sum()))
            theirs.add(tuple(heights))
        if mine != theirs:
            mism.append(s["observation_set_id"])
    check("repo enumerator == core.ground_truth", not mism, f"{len(samples)} sets, {len(mism)} mismatched")

    # 4 ---------------------------------------------------------------
    worst = 1.0
    for s in samples:
        cells = core.sample_cells(s)
        gt = core.ground_truth(cells)
        sc = core.score(list(gt), cells, gt, len(gt))
        worst = min(worst, sc["V"], sc["U"], sc["R"])
    check("slate at |GT| slots emits exactly GT", worst == 1.0, f"min(V,U,R)={worst:.3f}")

    # 5 ---------------------------------------------------------------
    # Prefer the recorded arm if it is on disk; otherwise draw the same i.i.d. arm here, so
    # the check still runs in a fresh clone where results/ (gitignored) does not exist.
    result_file = HERE.parent / "results" / "bench" / "iid_all129_kgt.json"
    by_gt: dict[int, list[float]] = {}
    source = "recorded arm"
    if result_file.exists():
        rows = json.loads(result_file.read_text(encoding="utf-8"))
        for r in rows["per_sample"]:
            by_gt.setdefault(r["n_gt"], []).append(r["R"])
    else:
        import random as _random

        rng = _random.Random(0)
        source = "recomputed here"
        for s in samples:
            cells = core.sample_cells(s)
            gt = list(core.ground_truth(cells))
            m = len(gt)
            picks = [gt[rng.randrange(m)] for _ in range(m)]
            by_gt.setdefault(m, []).append(len(set(picks)) / m)
    bad = []
    for m, rs in sorted(by_gt.items()):
        theory = 1 - (1 - 1 / m) ** m
        got = sum(rs) / len(rs)
        se = (theory * (1 - theory) / (len(rs) * m)) ** 0.5
        if abs(got - theory) > 4 * se:
            bad.append((m, got, theory))
    # the per-sample R is a single trial, so compare against the pooled mean with
    # the binomial SE of |GT| trials per sample; 4 sigma is deliberately loose.
    check("i.i.d. at |GT| slots matches 1-(1-1/m)^m", not bad, f"{source}: {bad}")

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
