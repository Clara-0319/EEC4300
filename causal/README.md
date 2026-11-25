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

LLM, 6 paired samples at k = 20: baseline 120 calls / 323 k tokens / $0.33 with R/cap = 0.983;
C3-Gen 6 calls / 39 k tokens / $0.04 with R/cap = **1.000** (7.4x cheaper, 20x fewer calls). Six
samples make this directional; the cost ratio holds on every one of them.

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

Model `deepseek-flash`, temperature 0.7. It is verbose when asked for many graphs (5-16 k
completion tokens per call) and samples with small `|GT|` are the expensive ones, so probe with
`--limit 1` first. `RetryingLLM` retries 6 times with backoff and logs to
`causal/results/bench/logs/api_errors.log`; it raises instead of silently recording an empty answer.

## Determinism

`run_experiment.py` re-executes itself under `PYTHONHASHSEED=0` so set iteration order cannot leak
into the numbers, and `core.enumerate_all` is a pure function of the observations. Every zero-cost
arm reproduces to the last digit (random-arm seed sd ~ 0.003). LLM arms are single-run.
