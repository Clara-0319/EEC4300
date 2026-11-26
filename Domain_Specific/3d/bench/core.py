"""G-Slate core: gravity-constrained 3D blocks as a product lattice.

Reduction theorem
-----------------
A structure lives on a GRID x GRID board with column heights h in 1..H for the
columns it occupies, and no block elsewhere (gravity: a block at height z needs a
block at z-1 in the same column, so a column is a contiguous stack from z=1).

The observation is the TOP VIEW only: a 0/1 matrix that is the OR of all layers.
Therefore:

    top view == support of the heights
    compatible structures == {1..H}^k  where k = number of occupied columns

so |GT| = H^k and "reconstruct the structure" is not a search over layer grids,
it is a choice of k integers.  Validation reduces to: the emitted structure
occupies exactly the observed columns and every height is in 1..H.

Consequence (i.i.d. ceiling)
----------------------------
A sampler that draws |GT| = H^k structures uniformly with replacement covers
1 - (1 - 1/|GT|)^|GT| in expectation, which is 0.7037 / 0.6536 / 0.6390 for
|GT| = 3 / 9 / 27 and 0.6476 weighted over the 129 observation sets of the
shipped 3x3x3 configuration.  No amount of "understanding 3D" moves that
number; only avoiding duplicates does.

A canonical structure is a tuple of heights, ordered by the row-major order of
the occupied cells.  This module never imports the benchmark; it is standalone.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

GRID = 3
MAX_HEIGHT = 3

Heights = Tuple[int, ...]


# ---------------------------------------------------------------- geometry

def cells_of(top_view: str, grid: int = GRID) -> List[Tuple[int, int]]:
    """Occupied cells in row-major order for a row-major 0/1 string."""
    cells = []
    for idx, ch in enumerate(top_view):
        if ch == "1":
            cells.append((idx // grid, idx % grid))
    return cells


def top_view_of(cells: Sequence[Tuple[int, int]], grid: int = GRID) -> str:
    chars = ["0"] * (grid * grid)
    for r, c in cells:
        chars[r * grid + c] = "1"
    return "".join(chars)


def ground_truth(cells: Sequence[Tuple[int, int]], max_height: int = MAX_HEIGHT) -> List[Heights]:
    """All compatible structures: the product set {1..H}^k."""
    return [tuple(h) for h in itertools.product(range(1, max_height + 1), repeat=len(cells))]


def is_valid(heights: Sequence[int], cells: Sequence[Tuple[int, int]], max_height: int = MAX_HEIGHT) -> bool:
    if len(heights) != len(cells):
        return False
    return all(isinstance(h, int) and 1 <= h <= max_height for h in heights)


def to_layers(cells: Sequence[Tuple[int, int]], heights: Sequence[int],
              max_height: int = MAX_HEIGHT, grid: int = GRID) -> List[List[List[int]]]:
    """Render a structure as layer grids (bottom first) for the paper's format."""
    layers = [[[0] * grid for _ in range(grid)] for _ in range(max_height)]
    for (r, c), h in zip(cells, heights):
        for z in range(h):
            layers[z][r][c] = 1
    while layers and not any(any(row) for row in layers[-1]):
        layers.pop()
    return layers


# ---------------------------------------------------------------- dispersion

def cheb(a: Sequence[int], b: Sequence[int]) -> int:
    """Chebyshev distance between two height vectors."""
    return max(abs(x - y) for x, y in zip(a, b))


def greedy_slate(points: Sequence[Heights], k: int, seed: int = 0) -> List[Heights]:
    """Farthest-first (max-min) selection on the height lattice."""
    rng = random.Random(seed)
    start = points[rng.randrange(len(points))]
    chosen = [start]
    while len(chosen) < k:
        best, best_d = None, -1
        for p in points:
            if p in chosen:
                continue
            d = min(cheb(p, c) for c in chosen)
            if d > best_d:
                best, best_d = p, d
        chosen.append(best)
    return chosen


def min_pair_distance(chosen: Iterable[Heights]) -> int:
    pairs = list(itertools.combinations(list(chosen), 2))
    if not pairs:
        return 0
    return min(cheb(a, b) for a, b in pairs)


def optimal_slate(points: Sequence[Heights], k: int, max_combos: int = 300_000):
    """Exact max-min slate by brute force; returns (min_distance, chosen) or None.

    Only used to audit that the greedy heuristic is optimal on this domain.
    """
    n = len(points)
    if k > n or math.comb(n, k) > max_combos:
        return None
    best = (-1, None)
    for combo in itertools.combinations(points, k):
        d = min_pair_distance(combo)
        if d > best[0]:
            best = (d, list(combo))
    return best


# ---------------------------------------------------------------- scoring

def score(emitted: Sequence[Sequence[int]], cells: Sequence[Tuple[int, int]],
          gt: Sequence[Heights], k_slots: int, max_height: int = MAX_HEIGHT) -> Dict[str, float]:
    """The paper's three metrics, with the structural cap made explicit.

    V = valid / k_slots            (k_slots = number of queries = slots)
    U = unique_valid / k_slots
    R = recovered / |GT|
    cap = min(k_slots, |GT|) / |GT|, which is the largest R reachable by any
    method that emits k_slots distinct structures.
    """
    emitted = [tuple(e) if not isinstance(e, tuple) else e for e in emitted]
    valid = [e for e in emitted if is_valid(e, cells, max_height)]
    uniq = list(dict.fromkeys(valid))
    gt_set = set(gt)
    recovered = [u for u in uniq if u in gt_set]
    n_gt = len(gt_set)
    denom = k_slots if k_slots else 1
    R = len(recovered) / n_gt if n_gt else 0.0
    cap = min(k_slots, n_gt) / n_gt if n_gt else 0.0
    return {
        "V": len(valid) / denom,
        "U": len(uniq) / denom,
        "R": R,
        "cap": cap,
        "R_cap": (R / cap) if cap else 0.0,
        "n_gt": n_gt,
        "n_emitted": len(emitted),
        "purity": (len(valid) / len(emitted)) if emitted else 0.0,
        "fill": len(emitted) / denom,
        "n_distinct": len(uniq),
    }


# ---------------------------------------------------------------- dataset

def build_dataset(grid: int = GRID, max_height: int = MAX_HEIGHT, max_blocks: int = 3) -> Dict:
    """All observation sets of the shipped configuration: 129 = C(9,1)+C(9,2)+C(9,3)."""
    samples = []
    for n_blocks in range(1, max_blocks + 1):
        for idx, comb in enumerate(itertools.combinations(range(grid * grid), n_blocks)):
            cells = [(i // grid, i % grid) for i in comb]
            gt = ground_truth(cells, max_height)
            samples.append({
                "observation_set_id": f"b{n_blocks}_{idx:03d}",
                "n_blocks": n_blocks,
                "top_view": top_view_of(cells, grid),
                "cells": [[r, c] for r, c in cells],
                "n_compatible_structures": len(gt),
                "ground_truth_structures": [list(h) for h in gt],
            })
    total = sum(s["n_compatible_structures"] for s in samples)
    return {
        "metadata": {
            "grid": grid,
            "max_height": max_height,
            "max_blocks": max_blocks,
            "n_observation_sets": len(samples),
            "hypothesis_space_size": total,
            "by_n_blocks": {
                str(k): sum(1 for s in samples if s["n_blocks"] == k)
                for k in range(1, max_blocks + 1)
            },
            "generation_method": "exhaustive_enumeration",
            "complete_gt_coverage": True,
            "observation_type": "top",
        },
        "samples": samples,
    }


def load_dataset(path: str | Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["samples"]


def sample_cells(sample: Dict) -> List[Tuple[int, int]]:
    return [tuple(c) for c in sample["cells"]]
