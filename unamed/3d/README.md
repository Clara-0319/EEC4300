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

## LLM runs (20 paired samples, thinking disabled)

Settings: 20 paired samples (identical ids for both arms), paper budget `k = |GT|`,
`thinking: {"type": "disabled"}` (chain of thought off; `run_llm.py --thinking disabled`),
temperature 0.7. The 20 ids are 7 with one occupied column, 7 with two, 6 with three.

| Arm | Prompt | Validity | Uniqueness | Recovery | R/cap | Cost |
|---|---|---|---|---|---|---|
| A0 | paper prompt (layer grid) | .920 | .804 | .804 | .804 | 2.06 CNY |
| A2 | height table + explicit "already proposed" list | 1.000 | **1.000** | **1.000** | **1.000** | **0.18 CNY** |

Paired over the 20 ids: A2 - A0 = **+0.196**, bootstrap 95 % CI **[+0.107, +0.307]**,
Wilcoxon **p = 0.0014**, wins / ties / losses **13 / 7 / 0** (A2 is never worse). A2 costs 11x less.

By number of occupied columns (R/cap, A0 vs A2): 1 column (n = 7) .857 / 1.000;
2 columns (n = 7) .857 / 1.000; 3 columns (n = 6) **.679** / 1.000.

Three findings, each reproducible from the result files:

1. **The paper prompt loses format compliance.** 8 of the 20 answers fail to parse at least once
   (`purity = .920` overall, and .857 on the single-column samples), because the layer-grid ASCII
   format is not reproduced reliably without a chain of thought. The height-table format (A2) parses
   on 20 / 20. The bottleneck is the output representation before it is the search.
2. **What A0 does get comes from the easy samples.** In the only discriminating bucket (`|GT|` = 27)
   it scores **.679**, far from 1.000, and that bucket is the entire remaining gap.
3. **The format alone is enough.** A2 reaches the ceiling `1.000 / 1.000 / 1.000` on 20 / 20
   samples with the exact enumerator never called (`n_filled = 0`), at 1/10 of the layer-grid cost.
   Diversity is carried by the explicit "already proposed" list in the prompt, not produced by the
   model: the same scheme without that list collapses to one answer per sample.
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

python 3d/bench/run_llm.py --arm schema_dedupe --limit 20 --budget-cny 0.6 --thinking disabled
python 3d/bench/run_llm.py --arm paper --limit 20 --budget-cny 2.0
```

`--budget-cny` is checked before each sample and after each call, and every finished sample is
flushed to disk immediately, so `--resume <file>` continues without paying twice for the same
sample ids. Model `deepseek-flash`, temperature 0.7; probe with `--limit 1` first.

## Determinism

The dataset and both zero-cost arms are deterministic (no seed is involved: the space is
enumerated, and i.i.d. sampling uses a fixed `random.Random(0)` in `selfcheck.py` / its own seed in
`run_free.py`). LLM arms are single-run and only reported as directional, with their sample counts.
