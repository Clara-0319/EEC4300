# Causal graph inference bench 

Additions to this repository for the **causal graph inference** domain of the HypoSpace benchmark.
One method — **C3-Gen** (constraint compilation -> constrained generation -> minimal repair ->
coverage slate) — plus a provable ceiling on what any method can score, a reproducible n=6 dataset,
and a set of deterministic + LLM comparison runs.

Nothing here modifies existing code. This package only *reads* `causal/run_causal_benchmark.py`
to reuse the paper's prompt and parser byte-for-byte (verified by `selfcheck.py`).

## Quick start

```bash
# run from the repository root
python causal/bench/selfcheck.py          # 5 checks, exits non-zero on failure
python causal/bench/run_experiment.py --arm oracle --k 5
python causal/bench/run_experiment.py --arm random --k 5
```

Results go to `causal/results/bench/{arm}_{dataset}_{tag}_k{k}.json` (gitignored).
`python causal/bench/report.py` prints the assembled table and re-verifies the `R/cap == U`
identity on every stored sample row; `--write` refreshes `causal/results/bench/REPORT.md`.

Inputs that are absent are reported as `[SKIP]`, never as `[FAIL]`. The n=5 dataset is 135 MB and
is not in the repository, so a fresh clone skips exactly the two n=5 checks and runs the other
three on n=6.

## Arms

| arm | what it does | cost |
|---|---|---|
| `oracle` | exact enumeration over the constraint whitelist, then farthest-first slate | 0 |
| `symbolic` | same search with a state budget (reproduces the "CCGIS" style search) | 0 |
| `random` | uniform sampling of valid graphs from the whitelist (the -LLM ablation) | 0 |
| `llm_baseline` | the paper's prompt: one graph per call, k calls | API |
| `c3gen` | constrained prompt asking for k graphs at once, with repair and critique retry | API |
| `c3gen_hybrid` | `c3gen`, then fill leftover slots from exact enumeration | API |

## Results, n = 6, 120 sets

| Method | k | Validity | Uniqueness | Recovery | R/cap |
|---|---|---|---|---|---|
| constrained enumeration | 5 | .948 | .948 | .274 | **1.000** |
| random sampling | 5 | .833 | .833 | .195 | .873 |
| constrained enumeration | 20 | .824 | .824 | .566 | **1.000** |
| random sampling | 20 | .605 | .352 | .352 | .702 |
| constrained enumeration | 50 | .653 | .653 | .823 | **1.000** |
| random sampling | 50 | .377 | .463 | .463 | .568 |

On the `|GT| <= k` subset ("must find them all", 12.5 % of the set at k = 5) complete search scores
R/cap 1.000 against 0.627 / 0.495 / 0.556 for random sampling at k = 5 / 20 / 50.

## Results on the LLM (`deepseek-flash`, temperature 0.7)

Settings: 20 paired samples (identical ids for both arms), `k` = 20,
`thinking: {"type": "disabled"}` (chain of thought off; reproducible with `--thinking disabled`),
temperature 0.7. Same model, same samples, same `k`; only the prompt and the repair step differ.

| Arm | Validity | Uniqueness | Recovery | R/cap | Calls | Tokens | Cost |
|---|---|---|---|---|---|---|---|
| `llm_baseline` (paper prompt) | .412 | .205 | .375 | .478 | 400 | 614 k | $0.55 |
| `c3gen` | .497 | .497 | .449 | **.776** | 39 | 39 k | **$0.03** |

Paired over the same 20 sample ids: mean difference **+0.2975**, bootstrap 95 % CI
**[+0.108, +0.478]**, Wilcoxon **p = 0.026**, wins / ties / losses **10 / 6 / 4**. Cost is **20.7x**
lower and calls **10.3x** fewer.

By |GT| stratum (R/cap, baseline vs C3-Gen): 1 (n = 4) 1.000 / 1.000; 5-16 (n = 4) .592 / .617;
17-64 (n = 6) .333 / **.675**; 65-256 (n = 6) .200 / **.833**. The gap opens only once the search
space is large: with a thin ground truth the baseline hits the cap by luck.

How to read the two `V` values:

- `Validity = fill x purity`. The baseline fills 93 % of its slots but only 45 % of what it emits is
  a valid graph (`purity = .452`); C3-Gen fills 50 % of its slots and everything it emits is valid
  (`purity = 1.000`). The comparison that matters is `Uniqueness` / `R/cap`.
- `U = .497` is the structural ceiling here, not a shortfall: 8 of the 20 samples have `|GT| < 20`,
  so 20 distinct compatible graphs do not exist for them and no method can score higher.
## Two facts that decide how these numbers should be read

1. **`Recovery = min(k, |GT|) / |GT|` is a structural ceiling.** Any method emitting `k` distinct
   valid graphs hits it exactly, because validity means membership in the ground truth. A "100 %
   recovery" on a small-|GT| sample says nothing about the method; report `k`, `|GT|` and
   `R/cap = Recovery / cap` together.
2. **`R/cap` equals `Uniqueness` whenever `|GT| >= k`** (108 of 120 sets at k = 5), so coverage
   carries no extra information outside the `|GT| <= k` regime, where it measures completeness.

At n = 5 the search space has 29 281 graphs and random sampling already reaches R/cap = 0.928
against 1.000 for complete search: the domain is saturated there and cannot separate methods. The
n = 6 space has 3 781 503 graphs and does separate them.

## Data

`causal/datasets/node06/n6_hard.json` (3.0 MB, 120 sets, seed 33550336, `gt_cap = 200`) is
force-added because `.gitignore` excludes `datasets/` and `*.json`, and because a clone should run
without a generation step. It is byte-identical to what the generator produces:

```bash
python causal/bench/make_dataset.py --out causal/datasets/node06/n6_hard.json   # SHA256 stable
```

`manifest_n6_all120.json` is the full sample list; `manifest_n6.json` is the older 72-sample
stratified subset kept for comparison. The n=5 dataset is not included (135 MB, above GitHub's file
limit); it comes from the repository's own `causal/generate_causal_dataset_for_large.py` (pass
`--seed` explicitly — that script fixes no seed by default).

## LLM arms (optional, paid)

```bash
export DEEPSEEK_API_KEY=sk-...          # or llm.api_keys.deepseek in causal/config/config_gpt4o.yaml

python causal/bench/run_experiment.py --arm llm_baseline --k 5 \
    --dataset causal/datasets/node06/n6_hard.json \
    --manifest causal/datasets/node06/manifest_n6_all120.json
```

Model `deepseek-flash`, temperature 0.7. Add `--thinking disabled` to turn the chain of thought
off; without the flag the server default (enabled) is used and the run is recorded as such in the
result file. It is verbose when asked for many graphs (5-16 k completion tokens per call) and samples
with small `|GT|` are the expensive ones, so probe with `--limit 1` first. `RetryingLLM` retries 6
times with backoff and logs to `causal/results/bench/logs/api_errors.log`; it raises instead of
silently recording an empty answer.

## Determinism

`run_experiment.py` re-executes itself under `PYTHONHASHSEED=0` so set iteration order cannot leak
into the numbers, and `core.enumerate_all` is a pure function of the observations. Every zero-cost
arm reproduces to the last digit (random-arm seed sd ~ 0.003). LLM arms are single-run.
