"""Build the 129-observation-set 3D dataset (shipped configuration: 3x3, max height 3, <=3 blocks).

Deterministic and exhaustive: the sample set is every subset of 1..3 occupied
cells of a 3x3 board, and every ground truth is the full product set {1..3}^k.
No random seed is involved, so re-running always reproduces the same file.

    python make_dataset.py --out ../datasets/node3x3_h3_all129.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "datasets" / "node3x3_h3_all129.json"))
    ap.add_argument("--grid", type=int, default=core.GRID)
    ap.add_argument("--max-height", type=int, default=core.MAX_HEIGHT)
    ap.add_argument("--max-blocks", type=int, default=3)
    args = ap.parse_args()

    data = core.build_dataset(args.grid, args.max_height, args.max_blocks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    md = data["metadata"]
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"observation sets : {md['n_observation_sets']}")
    print(f"total structures : {md['hypothesis_space_size']}")
    print(f"by n_blocks      : {md['by_n_blocks']}")
    sizes = sorted({s["n_compatible_structures"] for s in data["samples"]})
    print(f"|GT| values      : {sizes}")


if __name__ == "__main__":
    main()
