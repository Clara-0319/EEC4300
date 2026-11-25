"""Self-checks for the claims the report rests on.

    python causal/bench/selfcheck.py        # exit code 0 iff every check passes

Nothing here calls an API.  Each check exists because a specific number in the report
depends on it, so a failure means the report is wrong, not just the code.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import random
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
CAUSAL_DIR = BENCH_DIR.parent
for p in (str(BENCH_DIR), str(CAUSAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arms  # noqa: E402
import core  # noqa: E402
import data  # noqa: E402

DATASETS = [
    CAUSAL_DIR / "datasets" / "node05" / "n5_sampled.json",
    CAUSAL_DIR / "datasets" / "node06" / "n6_hard.json",
]
MANIFESTS = [
    CAUSAL_DIR / "datasets" / "node05" / "manifest_n60_seed33550336.json",
    CAUSAL_DIR / "datasets" / "node06" / "manifest_n6.json",
]
RESULTS = CAUSAL_DIR / "results" / "bench"


def _load(path):
    nodes, meta, raw = data.load_dataset(path)
    return [data.to_sample(r, nodes, meta.get("max_edges"), seed=0) for r in raw]


def _paper_prompt_builder():
    spec = importlib.util.spec_from_file_location(
        "rcb", str(CAUSAL_DIR / "run_causal_benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CausalBenchmarkEnhanced


def check_enumeration(paths):
    """Exhaustive search over the whitelist must reproduce the stored ground truth.

    Three separate failures are possible and all three invalidate coverage numbers:
    the stored count disagrees with the stored list, a stored graph does not actually
    satisfy the observations, or enumeration disagrees with the stored list.  The
    second one is how a direction-destroying bug in the dataset writer survives a
    completeness test: `sorted(sorted(edge))` turns C->B into B->C.
    """
    bad = []
    total = 0
    for path in paths:
        for r in _load(path):
            total += len(r["gt"])
            if len(r["gt"]) != r["gt_declared"]:
                bad.append((Path(path).name, r["id"], "declared", len(r["gt"]), r["gt_declared"]))
                continue
            if any(not core.validate(r["nodes"], g, r["constraints"]) for g in r["gt"]):
                bad.append((Path(path).name, r["id"], "invalid-gt"))
                continue
            got = {core.graph_key(g) for g in core.enumerate_all(r["constraints"])}
            if got != {core.graph_key(g) for g in r["gt"]}:
                bad.append((Path(path).name, r["id"], "enumeration", len(got), len(r["gt"])))
    return (not bad), f"{len(paths)} datasets, {total} stored graphs, problems={bad[:5]}"


def check_prompt(manifests):
    """The baseline arm must send the paper's prompt byte-for-byte."""
    builder = _paper_prompt_builder()
    bad = []
    n = 0
    for m in manifests:
        for r in _load_from_manifest(m):
            n += 1
            b = builder.__new__(builder)
            b.nodes = r["nodes"]
            b.max_edges = r["max_edges"]
            paper = b.create_prompt(r["observations"], [])
            mine = arms.baseline_prompt(r["nodes"], r["observations"], [], r["max_edges"])
            if paper != mine:
                bad.append(r["id"])
    return (not bad), f"{n} samples, differing={bad[:5]}"


def _load_from_manifest(manifest_path):
    # the manifest records the dataset it was drawn from, as a path relative to the repo
    # root in older files; fall back to the known pairing when the path is stale
    man = json.load(open(manifest_path, encoding="utf-8"))
    dataset = Path(man["dataset"])
    if not dataset.exists():
        dataset = Path(man["dataset"].replace(chr(92) + chr(92), chr(92)))
    if not dataset.exists():
        dataset = {
            "manifest_n60_seed33550336.json": DATASETS[0],
            "manifest_n6.json": DATASETS[1],
        }[Path(manifest_path).name]
    return data.load_manifest_samples(manifest_path, dataset)


def check_validity_theorem(samples, rng, per_sample=20):
    """Check the theorem against an independent, definition-level implementation.

    (a) soundness of the whitelist: in any whitelisted DAG the descendants of `s`
        contain no node the observation says must be 0, so forbidden reachability never
        needs its own check;
    (b) `validate(g)` agrees with brute force -- "descendants of the perturbed node
        equal the observed 1-set", for every observation.
    """
    def explains(nodes, g, observations):
        adj = core.adjacency(nodes, g)
        for o in observations:
            want = {x for x, v in o["effects"].items() if v == 1}
            if core.reach_from(adj, o["perturbed_node"]) != want:
                return False
        return True

    leaks, disagree, n_graphs = [], [], 0
    for r in samples:
        nodes, c = r["nodes"], r["constraints"]
        for _ in range(per_sample):
            g = arms._random_valid(nodes, c, rng, attempts=10)
            if g is None:
                continue
            n_graphs += 1
            adj = core.adjacency(nodes, g)
            for s in sorted(c.required):
                if core.reach_from(adj, s) & c.forbidden[s]:
                    leaks.append((r["id"], s))
            if core.validate(nodes, g, c) != explains(nodes, g, r["observations"]):
                disagree.append(r["id"])
    ok = not leaks and not disagree
    return ok, (f"{n_graphs} graphs; whitelist leaks={leaks[:3]} "
                f"validate/brute-force disagreements={disagree[:3]}")


def check_selection_irrelevant(samples, k=5, rng=None):
    """Which k valid graphs you emit cannot change recovery.

    Recovery counts emitted graphs that are in GT; validity means "is in GT"; so with
    |GT| >= k any k distinct valid graphs recover exactly k of them.  Diversity-selection
    heuristics therefore cannot move R in this regime -- measured, not assumed.
    """
    rng = rng or random.Random(0)
    n = bad = 0
    for r in samples:
        valid = core.enumerate_all(r["constraints"])
        gt = set(r["gt"])
        if len(valid) <= k or len(gt) < k:
            continue
        n += 1
        cap = min(k, len(gt)) / len(gt)
        picks = [
            core.farthest_first(valid, k, seed=0),
            [valid[i] for i in sorted(rng.sample(range(len(valid)), k))],
        ]
        for pick in picks:
            if abs(len(set(pick) & gt) / len(gt) - cap) > 1e-12:
                bad += 1
    return bad == 0, f"{n} samples with |GT|>{k}; strategies hitting the cap: {2*n-bad}/{2*n}"


def check_metric_identity():
    """R/cap == U whenever |GT| >= k, on every stored result file."""
    checked = viol = 0
    for f in sorted(glob.glob(str(RESULTS / "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        k = d["aggregate"]["k"]
        for r in d["per_sample"]:
            if "error" in r or r["n_gt"] < k:
                continue
            checked += 1
            if abs(r["R_norm"] - r["n_unique_valid"] / k) > 1e-9:
                viol += 1
    if checked == 0:
        return False, "no result files with |GT| >= k rows; run an arm first"
    return viol == 0, f"{checked} sample-rows, violations={viol}"


def main():
    rng = random.Random(12345)

    # A fresh clone has no datasets/ and no results/ (both are gitignored, and the n=5
    # file is 135 MB).  Missing inputs are reported as SKIP, never as FAIL: a check that
    # cannot run is not a check that failed.
    datasets = [p for p in DATASETS if Path(p).exists()]
    for p in DATASETS:
        if not Path(p).exists():
            print(f"[SKIP] dataset not present: {p}  (regenerate it, see causal/bench/README.md)")
    manifests = [p for p in MANIFESTS if Path(p).exists()]
    for p in MANIFESTS:
        if not Path(p).exists():
            print(f"[SKIP] manifest not present: {p}")

    manifest_samples = []
    usable_manifests = []
    for m in manifests:
        try:
            rows = _load_from_manifest(m)
        except Exception as exc:  # noqa: BLE001 - a manifest whose dataset is absent is a SKIP
            print(f"[SKIP] manifest {Path(m).name} unusable: {type(exc).__name__}: {exc}")
            continue
        usable_manifests.append(m)
        manifest_samples.extend(rows)

    has_results = RESULTS.exists() and next(RESULTS.glob("*.json"), None) is not None

    checks = []
    if datasets:
        checks.append(("enumerator == dataset ground truth", lambda: check_enumeration(datasets)))
    if usable_manifests:
        checks.append(("baseline prompt == paper prompt",
                       lambda: check_prompt(usable_manifests)))
    if manifest_samples:
        checks.append(("validity theorem both directions",
                       lambda: check_validity_theorem(manifest_samples, rng)))
    if has_results:
        checks.append(("R/cap == U identity", check_metric_identity))
    else:
        print("[SKIP] no result files yet: run an arm first, e.g. "
              "`python run_experiment.py --arm oracle --k 5`")
    if manifest_samples:
        checks.append(("selection heuristic cannot move R",
                       lambda: check_selection_irrelevant(manifest_samples, rng=rng)))

    failures = 0
    for name, fn in checks:
        ok, detail = fn()
        failures += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not checks:
        print("nothing to check: no dataset, manifest or result file present")
        return 0
    print("all checks passed" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
