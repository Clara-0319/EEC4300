# Gravity-constrained 3D voxel bench 

Additions to this repository for the **3D voxel reconstruction** domain of the HypoSpace
benchmark. One method — **G-Slate** (reduce the layer-stack problem to a height vector per column,
enumerate the product space, lay out a max-min-distance slate) — plus the closed-form ceiling for
any sampling method, the full 129-set dataset, and deterministic + LLM comparison runs.

Nothing here modifies existing code: `3d/run_3d_benchmark.py` is only imported (read-only) so the
paper's prompt and parser are reused byte-for-byte.

## Why the problem collapses

A structure on a 3x3 grid with at most three blocks in top view and maximum height 3 is fully
determined by the heights of its occupied columns: gravity forces every column to be filled from
the bottom up, so no other shape fits the same top view. An observation therefore only says *how
many* columns are occupied, and the ground truth is the entire product set

```
|GT| = 3^k          k = number of occupied columns (1, 2 or 3)
```

so the compatible structures are exactly `{1,2,3}^k`, and `core.ground_truth` enumerates them.
This is checked sample by sample against the repository's own
`enumerate_all_structures_by_heights`: 129 / 129 identical.

Given `|GT|` attempts (the paper's `--query-multiplier 1.0`), uniform sampling can only cover
`1 - (1 - 1/m)^m` of the ground truth in expectation — 0.704 / 0.654 / 0.639 for `m` = 3 / 9 / 27,
weighted **0.648** over the 129 sets. Half the coverage loss in this domain is therefore a
coupon-collector property of the task, not a model failure.

## Quick start

```bash
# run from the repository root
python 3d/bench/selfcheck.py                  # 5 checks, exits non-zero on failure
python 3d/bench/run_free.py --arm both --budget gt    # paper budget, zero cost
python 3d/bench/run_free.py --arm both --budget 5     # k = 5 slots
```

Results go to `3d/results/bench/{arm}_all129_k{gt|5}.json` (gitignored).

## Results, 129 sets

Paper budget, `k = |GT|`:

| Method | Validity | Uniqueness | Recovery |
|---|---|---|---|
| i.i.d. sampling | 1.000 | .654 | .654 |
| G-Slate | 1.000 | **1.000** | **1.000** |

Fixed budget, `k = 5` slots: i.i.d. reaches R = .303 (cap .345, R/cap .902); G-Slate reaches
R = **.345** (cap .345, R/cap **1.000**) with a minimum pairwise Chebyshev distance of 1.68.

`greedy_slate` is not merely a heuristic: on `{1,2,3}^k` for k = 1, 2, 3 it matches the exhaustive
optimum of the min-distance objective for budgets 3 and 5.

## LLM runs (paid, 20 paired samples, ~3.85 CNY)

| Arm | Prompt | Validity | Uniqueness | Recovery | Cost |
|---|---|---|---|---|---|
| A0 | paper prompt (layer grid) | 1.000 | .883 | .883 | 2.58 CNY |
| A1 | height table, no history | 1.000 | .167 | .167 | 0.42 CNY |
| A2 | height table + explicit "already proposed" list | 1.000 | **1.000** | **1.000** | 0.85 CNY |

Three findings, each reproducible from the result files:

1. A0's 0.883 comes entirely from the easy samples. In the only discriminating bucket (`|GT|` = 27)
   it scores **0.611**, *below* the 0.639 that plain sampling with duplicates already achieves —
   i.e. with a history list in the prompt the model behaves like a die with repeats.
2. Remove the history list (A1) and diversity collapses to exactly one answer per sample
   (`U = 1/k`): diversity is carried by the prompt's prior list, not produced by the model.
3. The format is not the bottleneck: A2 on the same samples reaches the ceiling 20 / 20 with the
   exact enumerator never called (`n_filled = 0`). The height-table prompt also costs 3x less than
   the layer-grid prompt at equal-or-better coverage.

## Data

`3d/datasets/node3x3_h3_all129.json` (136 KB) contains all 129 observation sets — every 3x3 top
view with 1, 2 or 3 occupied columns — with their ground-truth height vectors. It is force-added
because `.gitignore` excludes `datasets/` and `*.json`. It is deterministic and reproducible:

```bash
python 3d/bench/make_dataset.py     # rewrites the same 129 sets, no seed involved
```

## LLM runs: budget controls

```bash
export DEEPSEEK_API_KEY=sk-...

python 3d/bench/run_llm.py --arm schema_dedupe --limit 20 --budget-cny 1.2
```

`--budget-cny` is checked before each sample and after each call, and every finished sample is
flushed to disk immediately, so `--resume <file>` continues without paying twice for the same
sample ids. Model `deepseek-flash`, temperature 0.7; probe the real price first with
`--limit 1 --budget-cny 0.05`.

## Determinism

The dataset and both zero-cost arms are deterministic (no seed is involved: the space is
enumerated, and i.i.d. sampling uses a fixed `random.Random(0)` in `selfcheck.py` / its own seed in
`run_free.py`). LLM arms are single-run and only reported as directional, with their sample counts.
