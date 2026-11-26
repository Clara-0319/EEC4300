"""Dataset loading + the frozen evaluation manifest.

The manifest is the single most important piece of experimental hygiene here: every
arm is scored on exactly the same samples, and the file is committed to the repo so a
later run cannot silently change the sample set (which is how the baseline and CCGIS
numbers ended up being computed on different data).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import core

BUCKET_EDGES = [1, 2, 5, 17, 65, 257]  # |GT| strata: [1], [2,4], [5,16], [17,64], [65,256], [257,inf)


def gt_bucket(n: int) -> str:
    for i in range(len(BUCKET_EDGES) - 1, -1, -1):
        if n >= BUCKET_EDGES[i]:
            if i == 0:
                return "1"
            hi = "" if i == len(BUCKET_EDGES) - 1 else f"-{BUCKET_EDGES[i + 1] - 1}"
            return f"{BUCKET_EDGES[i]}{hi}"
    return "1"


def load_dataset(path: str | Path) -> Tuple[List[str], dict, List[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    nodes = meta.get("nodes", [])
    sets: List[dict] = []
    if "datasets_by_n_observations" in data:
        for v in data["datasets_by_n_observations"].values():
            sets.extend(v)
    elif "datasets" in data:
        sets = data["datasets"]
    elif "sampled_datasets" in data:
        sets = data["sampled_datasets"]
    return nodes, meta, sets


def to_sample(raw: dict, nodes: Sequence[str], max_edges, seed: int) -> dict:
    observations = raw["observations"]
    gt = [frozenset((e[0], e[1]) for e in g["edges"]) for g in raw["ground_truth_graphs"]]
    return {
        "id": raw["observation_set_id"],
        "nodes": list(nodes),
        "max_edges": max_edges,
        "observations": observations,
        "gt": gt,
        "constraints": core.Constraints.from_observations(nodes, observations),
        "seed": seed,
        "n_obs": len(observations),
        "gt_file": len(gt),
        "gt_declared": raw.get("n_compatible_graphs"),
        "bucket": gt_bucket(len(gt)),
    }


def build_manifest(dataset_path, out_path, per_bucket: int = 10, seed: int = 33550336) -> dict:
    nodes, meta, sets = load_dataset(dataset_path)
    rng = random.Random(seed)
    by_bucket: Dict[str, List[dict]] = {}
    for s in sets:
        by_bucket.setdefault(gt_bucket(len(s["ground_truth_graphs"])), []).append(s)

    picked: List[dict] = []
    for bucket in sorted(by_bucket, key=lambda b: int(b.split("-")[0])):
        pool = sorted(by_bucket[bucket], key=lambda s: s["observation_set_id"])
        take = min(per_bucket, len(pool))
        chosen = rng.sample(pool, take) if take < len(pool) else list(pool)
        for s in chosen:
            picked.append({
                "id": s["observation_set_id"],
                "bucket": bucket,
                "n_obs": len(s["observations"]),
                "gt_size": len(s["ground_truth_graphs"]),
            })

    picked.sort(key=lambda x: x["id"])
    manifest = {
        "dataset": str(dataset_path),
        "nodes": nodes,
        "max_edges": meta.get("max_edges"),
        "seed": seed,
        "per_bucket": per_bucket,
        "n_samples": len(picked),
        "buckets": {b: sum(1 for s in sets if gt_bucket(len(s["ground_truth_graphs"])) == b)
                    for b in by_bucket},
        "samples": picked,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def load_manifest_samples(manifest_path, dataset_path, seed: int = 0) -> List[dict]:
    nodes, meta, sets = load_dataset(dataset_path)
    by_id = {s["observation_set_id"]: s for s in sets}
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)
    out = []
    for entry in man["samples"]:
        raw = by_id[entry["id"]]
        out.append(to_sample(raw, nodes, meta.get("max_edges"), seed))
    return out
