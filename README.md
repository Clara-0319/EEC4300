# Structured Hypothesis Generation for HypoSpace

Inference-time machine-learning methods for improving large language model (LLM) performance on **HypoSpace**, a benchmark for set-valued hypothesis generation under underdetermination.

- [HypoSpace paper](https://proceedings.mlr.press/v306/chen26bz.html)
- [Original HypoSpace benchmark](https://github.com/CTT-Pavilion/_HypoSpace)

## Overview

Many scientific inference problems admit multiple distinct hypotheses that are equally consistent with the available observations. In this setting, a useful model must do more than produce one valid answer: it must explore the admissible hypothesis space without repeatedly returning equivalent solutions.

HypoSpace evaluates LLMs as samplers over finite hypothesis spaces using **Validity**, **Uniqueness**, and **Recovery**. Its three structured domains are causal graph inference, gravity-constrained 3D voxel reconstruction, and Boolean genetic-interaction modeling. The benchmark exposes a common failure mode: Validity can remain high while Uniqueness and Recovery deteriorate as the solution space grows.

This repository investigates two complementary ways to address that problem:

| Research line | Scope | Main idea |
|---|---|---|
| [`Meta_Structure/`](Meta_Structure/) | General, shared across all three tasks | Guide generation with observation-independent structural primitives, explicit composition rules, and complexity-stratified sampling. |
| [`Domain_Specific/`](Domain_Specific/) | Specialized for each task | Exploit each domain's exact constraints and representation to improve validity, deduplication, repair, and coverage. |

The methods operate at **inference time**. They do not require fine-tuning or replacing the underlying LLM.

## Benchmark tasks

| Task | Input observations | Generated hypothesis | Structural challenge |
|---|---|---|---|
| Causal graph inference | Perturbation effects over system nodes | A directed acyclic graph consistent with every intervention | Many edge sets can induce the same reachability observations. |
| 3D voxel reconstruction | A top-view projection | A gravity-valid 3D structure | Different column heights can produce the same projection. |
| Boolean expression discovery | Partial input-output examples | A bounded Boolean expression | Syntactically different expressions may be structurally equivalent. |

## Method 1: Meta-structure-guided generation

`Meta_Structure/` provides a shared experimental framework for all three domains. Its central idea is to replace unconstrained, repeated answer generation with **structure-aware compositional generation**.

### Core components

1. **Fixed structural libraries**  
   Each task exposes a pre-generated, observation-independent library of primitives:
   - Boolean: bounded expressions that can be composed with an additional allowed gate.
   - 3D: vertical column blocks with explicit bottom-to-top semantics.
   - Causal: labeled directed-edge primitives plus an empty graph.

2. **Explicit composition protocol**  
   The model must return one complete hypothesis in the original benchmark format, not a list of primitive identifiers or an intermediate plan.

3. **Complexity-stratified decoding**  
   Requests can target a particular structural complexity: canonical operator count for Boolean expressions, occupied-voxel count for 3D structures, and distinct-edge count for causal graphs. This encourages coverage across regions of the hypothesis space rather than repeated sampling from a single mode.

4. **Deterministic validation and canonicalization**  
   Shared task adapters parse responses, check hypothesis-space membership, validate observations, and compute canonical keys for deduplication and recovery scoring.

The libraries do not contain reference answers and require no API calls to construct. Reference sets are used for evaluation and for deriving complexity schedules; they are removed from the public cases passed to prompt adapters.

### Comparison arms

| Arm | Purpose |
|---|---|
| `history_baseline` | Replicates the upstream prompt and prior-history sampling loop. |
| `paper_stratified` | Adds complexity-targeted requests without a structural library. |
| `prompt_library` | Adds fixed structural primitives without complexity stratification. |
| `stratified_library` | Combines structural primitives with complexity-targeted requests. |

The shared engine supports equal `M` and `2M` budgets, where `M` is the number of admissible reference hypotheses. It records per-call responses, metrics, token usage, and estimated cost, and audits completed runs against frozen datasets, libraries, source hashes, and protocol settings.

## Method 2: Domain-specific algorithms

### Causal graphs: C3-Gen

[`Domain_Specific/causal/`](Domain_Specific/causal/) implements **C3-Gen**, a four-stage pipeline:

1. **Constraint compilation:** convert perturbation observations into reachability constraints and an allowed-edge whitelist.
2. **Constrained generation:** ask the LLM for a complete slate of distinct graphs under the compiled constraints.
3. **Minimal repair:** correct malformed or invalid candidates instead of discarding the entire response.
4. **Coverage slate selection:** retain a diverse set of valid graphs and use critique or exact enumeration when slots remain unfilled.

The package also includes exact-enumeration, budgeted symbolic-search, random-sampling, paper-baseline, and hybrid arms. Its normalized coverage score, `R/cap`, helps distinguish method quality from the structural ceiling imposed when fewer than `k` valid hypotheses exist.

See the [causal methodology and results](Domain_Specific/causal/README.md).

### 3D reconstruction: G-Slate

[`Domain_Specific/3d/`](Domain_Specific/3d/) implements **G-Slate**, which uses gravity to simplify the hypothesis space. Every occupied column is filled from the bottom upward, so a structure can be represented by one height per occupied column.

For `k` occupied columns with maximum height 3, the admissible space is exactly `{1, 2, 3}^k`, with `|GT| = 3^k`. G-Slate enumerates this product space and constructs a slate using max-min-distance ordering rather than independent sampling.

The LLM comparison additionally evaluates a height-table representation with an explicit list of previously proposed structures. This addresses both output-format failures and duplicate generation.

See the [3D methodology and results](Domain_Specific/3d/README.md).

### Boolean expressions: validated feedback and structural archiving

[`Domain_Specific/boolean/`](Domain_Specific/boolean/) contains three incremental inference-time improvements:

1. **Valid unique-candidate archive:** validate each expression and archive only previously unseen mechanistic structures; provide targeted feedback for invalid or duplicate outputs.
2. **Validity-first feedback:** resolve constraint and observation violations before emphasizing structural novelty.
3. **Complete observation-mismatch diagnostics:** report every visible input on which a candidate fails rather than only the first mismatch.

All variants use structural deduplication based on commutativity, AND/OR idempotence, and associativity flattening. They aim to improve exploration while making the correctness-versus-diversity tradeoff explicit.

See the [Boolean methodology and results](Domain_Specific/boolean/README.md).

## Repository layout

```text
.
├── Meta_Structure/
│   ├── run_experiment.py          # Unified three-task entry point
│   ├── shared/                    # Prompt protocols, adapters, engine, logging, and audits
│   ├── boolean/                   # Preserved Boolean runner and task entry point
│   ├── causal/                    # Preserved causal runner and task entry point
│   └── 3d/                        # Preserved 3D runner and task entry point
└── Domain_Specific/
    ├── causal/                    # C3-Gen, datasets, comparison arms, and analysis
    ├── 3d/                        # G-Slate, datasets, comparison arms, and analysis
    └── boolean/                   # Baseline, three feedback variants, and saved results
```

## Getting started

### Prerequisites

Use a Python environment with the dependencies required by the selected task. Core numerical, graph, symbolic, and configuration imports include `numpy`, `networkx`, `scipy`, `sympy`, and `pyyaml`. LLM experiments additionally require access to a supported OpenAI-compatible provider.

Run the following commands from the repository root unless a task-specific document says otherwise.

### Prepare the general experiment without paid API calls

The general runner requires an explicit model ID. Without `--run`, it only generates and freezes local experiment inputs:

```bash
python Meta_Structure/run_experiment.py \
  --provider deepseek \
  --model deepseek-chat \
  --thinking disabled \
  --tasks boolean 3d causal \
  --difficulty medium hard \
  --methods history_baseline paper_stratified prompt_library stratified_library \
  --budget-multiplier 1 \
  --output outputs/meta-structure
```

The preparation stage reports the number of cases, runs, and planned calls, and makes zero API requests.

### Run the general experiment

Set credentials through an environment variable, never in source files or command-line arguments:

```bash
export DEEPSEEK_API_KEY="YOUR_API_KEY"
```

Then add `--run` to execute the frozen protocol:

```bash
python Meta_Structure/run_experiment.py \
  --provider deepseek \
  --model deepseek-chat \
  --thinking disabled \
  --output outputs/meta-structure \
  --run
```

To change tasks, methods, difficulty levels, budget, or parallelism, pass the corresponding options explicitly. To audit a completed experiment without further API calls:

```bash
python Meta_Structure/run_experiment.py \
  --output outputs/meta-structure \
  --report-only
```

### Run zero-cost domain-specific comparisons

```bash
# Exact causal enumeration with coverage-based slate selection
python Domain_Specific/causal/bench/run_experiment.py --arm oracle --k 5

# Independent sampling versus G-Slate at the paper's |GT| budget
python Domain_Specific/3d/bench/run_free.py --arm both --budget gt
```

For paid LLM runs, start with a small sample limit and consult the relevant task README for provider settings, budget controls, checkpointing, and resume behavior. Generated results and raw model exchanges are treated as local experiment artifacts.

## Evaluation metrics

| Metric | Meaning |
|---|---|
| **Validity** | Fraction of generated candidates that belong to the allowed hypothesis space and satisfy all visible observations. |
| **Uniqueness** | Fraction of generation slots occupied by distinct hypotheses after task-specific structural canonicalization. |
| **Recovery** | Fraction of the ground-truth hypothesis set recovered by valid, unique candidates. |
| **R/cap** | Recovery divided by the budget-dependent structural cap `min(k, |GT|) / |GT|`; used in the specialized causal and 3D analyses. |

Metrics should always be reported with the model, decoding settings, task subset, and query budget. Scores from different tasks or experimental budgets are not directly comparable.

## Reported results

The following values summarize the supplied `Meta_Structure` benchmark and the domain-specific task documentation. They use different models, samples, and budgets, so they should be read as separate comparisons rather than a single leaderboard.

### Meta_Structure: unified benchmark at the `1M` budget

| Model | Method | NR @ 1M | RR @ 1M |
|---|---|---:|---:|
| Qwen3.8-flash | History baseline | 9.23% | 7.47% |
| Qwen3.8-flash | Meta-structure | 22.04% | 9.23% |
| Qwen3.8-flash | Hybrid | **24.67%** | **12.32%** |
| DeepSeek-Flash | History baseline | 4.84% | 5.90% |
| DeepSeek-Flash | Meta-structure | 16.32% | 14.56% |
| DeepSeek-Flash | Hybrid | **24.25%** | **17.65%** |

Here, `NR` is the uniqueness (novelty) rate and `RR` is the recovery rate. At the `1M` budget, the number of generation slots equals `M`, the number of admissible reference hypotheses. Meta-structure improves both metrics over the history baseline for both models, while Hybrid achieves the highest reported values in each setting.

### Domain-specific causal: 20 paired samples, `k = 20`, `deepseek-flash`

| Method | Validity | Uniqueness | Recovery | R/cap | API calls |
|---|---:|---:|---:|---:|---:|
| Paper-prompt baseline | 0.412 | 0.205 | 0.375 | 0.478 | 400 |
| C3-Gen | 0.497 | 0.497 | 0.449 | **0.776** | **39** |

The paired mean `R/cap` improvement is **+0.2975**. The task report also provides bootstrap intervals, Wilcoxon testing, per-stratum results, and token-cost analysis.

### Domain-specific 3D: 20 paired samples, `k = |GT|`, thinking disabled

| Method | Validity | Uniqueness | Recovery | R/cap |
|---|---:|---:|---:|---:|
| Paper layer-grid prompt | 0.920 | 0.804 | 0.804 | 0.804 |
| Height table plus prior-proposal list | **1.000** | **1.000** | **1.000** | **1.000** |

Across all 129 deterministic cases at `k = |GT|`, G-Slate reaches Validity, Uniqueness, and Recovery of 1.000, compared with 0.654 Uniqueness and Recovery for the reported i.i.d. sampling arm.

### Domain-specific Boolean: three seeds, 30 samples per seed, 10 queries per sample

| Method | Validity | Uniqueness | Recovery |
|---|---:|---:|---:|
| Baseline | **93.78%** | 17.67% | 42.33% |
| Valid unique archive | 68.56% | **40.33%** | 74.67% |
| Validity-first feedback | 80.33% | 37.89% | **75.96%** |
| Complete mismatch diagnostics | 81.00% | 36.56% | 73.81% |

These results show a correctness-diversity tradeoff: the baseline has the highest Validity, while the feedback methods substantially improve structural exploration and Recovery.

## Reproducibility and experiment hygiene

- Fixed seeds and deterministic zero-cost arms are provided where applicable.
- General experiments freeze datasets, structural libraries, model settings, and source hashes before execution.
- Raw calls, response-level evaluations, learning curves, and token accounting are retained for audit.
- Completed general runs recompute metrics from raw responses and verify protocol consistency.
- Paid execution is opt-in through `--run`; local preparation and reporting do not call an API.
- API keys must remain in environment variables or ignored local configuration.
- Large or generated datasets and experiment outputs may be absent from a fresh clone; consult each task README for regeneration instructions and expected skipped checks.

## Citation

```bibtex
@InProceedings{pmlr-v306-chen26bz,
  title     = {{H}ypo{S}pace: A Diagnostic Benchmark for Set-Valued Hypothesis Generation under Underdetermination and Sublinear Coverage Bounds},
  author    = {Chen, Tingting and Lin, Beibei and Yuan, Zifeng and Zou, Qiran and He, Hongyu and Goyal, Anirudh and Ong, Yew-Soon and Liu, Dianbo},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {15451--15465},
  year      = {2026},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  url       = {https://proceedings.mlr.press/v306/chen26bz.html}
}
```
