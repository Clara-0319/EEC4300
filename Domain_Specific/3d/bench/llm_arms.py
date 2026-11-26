"""LLM arms for the 3D domain.

A0  `paper`   : the repository's own prompt (verbatim, via Benchmark3D.create_prompt)
                and its own parser -> maximum fidelity to the baseline.
A1  `schema`  : same information, but the answer is a height vector instead of
                layer grids ("Heights: 1 3 2").
A2  `schema_dedupe` : A1 with the already-emitted height vectors listed in the
                prompt, plus exact enumeration to fill slots the model leaves
                duplicated or unparsed (the same idea as c3gen_hybrid in causal).

Every arm emits exactly `k` slots; an unparsed answer occupies an invalid slot so
that Validity is not silently inflated.

Cost guard: `Budget` accumulates USD and raises when the CNY cap is reached.
"""

from __future__ import annotations

import importlib.util
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import core

BENCH_DIR = Path(__file__).resolve().parent
REPO_3D = BENCH_DIR.parent / "run_3d_benchmark.py"
USD_PER_CNY = 1 / 7.25
PRICE_IN, PRICE_OUT = 0.30, 1.20  # USD per 1M tokens, deepseek pricing

INVALID: Tuple = ("INVALID",)


def _load_repo():
    spec = importlib.util.spec_from_file_location("run3d", REPO_3D)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_REPO = None


def repo_bench():
    """A Benchmark3D instance good enough to build the paper's prompt."""
    global _REPO
    if _REPO is None:
        _REPO = _load_repo()
    b = _REPO.Benchmark3D.__new__(_REPO.Benchmark3D)
    b.metadata = {"max_height": core.MAX_HEIGHT}
    return b


# ---------------------------------------------------------------- client

class Budget:
    def __init__(self, cap_cny: float):
        self.cap_cny = cap_cny
        self.spent_usd = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage) -> float:
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        cost = pt / 1e6 * PRICE_IN + ct / 1e6 * PRICE_OUT
        self.spent_usd += cost
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        if self.spent_usd / USD_PER_CNY > self.cap_cny:
            raise RuntimeError(
                f"budget cap reached: spent {self.spent_usd / USD_PER_CNY:.2f} CNY > {self.cap_cny:.2f} CNY")
        return cost

    @property
    def spent_cny(self) -> float:
        return self.spent_usd / USD_PER_CNY


def make_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def ask(client, prompt: str, budget: Budget, model: str = "deepseek-flash",
        temperature: float = 0.7, attempts: int = 4,
        thinking: Optional[str] = None) -> str:
    extra = {"thinking": {"type": thinking}} if thinking else None
    last = None
    for i in range(attempts):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if extra:
                # DeepSeek: {"thinking": {"type": "disabled"}} turns chain-of-thought off
                kwargs["extra_body"] = extra
            resp = client.chat.completions.create(**kwargs)
            budget.add(resp.usage)
            return resp.choices[0].message.content or ""
        except RuntimeError:
            raise
        except Exception as e:  # network / 429 / 402
            last = e
            time.sleep(3 * (2 ** i))
    raise RuntimeError(f"api failed after {attempts} attempts: {last}")


# ---------------------------------------------------------------- conversion

def structure_to_heights(struct, cells: Sequence[Tuple[int, int]]) -> Tuple:
    """Structure3D -> height tuple over the observed cells, or INVALID."""
    try:
        import numpy as np
        arr = np.stack(struct.layers)
    except Exception:
        return INVALID
    if arr.ndim != 3:
        return INVALID
    heights = []
    for r, c in cells:
        col = arr[:, r, c]
        if not col.any():
            return INVALID          # the model left an occupied cell empty
        # gravity: the column must be a contiguous prefix of 1s
        ones = int(col.sum())
        if list(col[:ones]) != [1] * ones or any(col[ones:]):
            return INVALID
        heights.append(ones)
    # top view must match exactly: no blocks outside the observed cells
    top = arr.any(axis=0).astype(int)
    for r in range(top.shape[0]):
        for c in range(top.shape[1]):
            if top[r, c] and (r, c) not in set(cells):
                return INVALID
    return tuple(heights)


# ---------------------------------------------------------------- prompts

def prompt_a1(cells: Sequence[Tuple[int, int]], top_view: str,
              already: Sequence[Tuple[int, ...]] = ()) -> str:
    grid = core.GRID
    rows = "\n".join(" ".join(top_view[r * grid:(r + 1) * grid]) for r in range(grid))
    cell_txt = ", ".join(f"({r},{c})" for r, c in cells)
    p = f"""You are given the TOP VIEW of a 3D structure made of unit blocks on a 3x3 grid.
The maximum height is 3 layers.
A block at height z requires a block at height z-1 in the same column, so every occupied cell is a column of height 1, 2 or 3.

Top view (1 = a column exists at that cell):
{rows}

Occupied cells in row-major order: {cell_txt}
"""
    if already:
        prev = "; ".join("[" + " ".join(str(h) for h in a) + "]" for a in already)
        p += f"""
Already proposed (each is a valid structure; you must give a DIFFERENT one):
{prev}
"""
    p += f"""
Task: give the column heights of ONE structure compatible with the top view.

Output exactly one line and nothing else:

Heights: <{" ".join("h" + str(i + 1) for i in range(len(cells)))}>

where each h is an integer in 1..3, in the same order as the cells listed above.
"""
    return p


def parse_a1(text: str, n_cells: int) -> Tuple:
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("heights:"):
            line = line.split(":", 1)[1]
        nums = re.findall(r"\b([1-3])\b", line)
        if len(nums) == n_cells:
            return tuple(int(x) for x in nums)
    # fallback: last line that is exactly n_cells small integers
    for line in reversed(text.splitlines()):
        nums = re.findall(r"\b([1-3])\b", line)
        if len(nums) == n_cells and len(line.split()) <= n_cells + 2:
            return tuple(int(x) for x in nums)
    return INVALID


def prompt_a0(bench, top_view: str, priors: List) -> str:
    return bench.create_prompt(top_view, prior_structures=priors or None)


def parse_a0(bench, text: str, cells) -> Tuple:
    struct = bench.parse_llm_response(text)
    if struct is None:
        return INVALID
    return structure_to_heights(struct, cells)


# ---------------------------------------------------------------- arms

def run_arm(arm: str, sample: Dict, k: int, client, budget: Budget,
            rng_seed: int = 0, thinking: Optional[str] = None) -> Dict:
    cells = core.sample_cells(sample)
    top_view = sample["top_view"]
    gt = core.ground_truth(cells)
    bench = repo_bench() if arm == "paper" else None

    emitted: List[Tuple] = []
    priors: List = []
    n_llm = 0
    for _ in range(k):
        if arm == "paper":
            p = prompt_a0(bench, top_view, priors)
            text = ask(client, p, budget, thinking=thinking)
            struct = bench.parse_llm_response(text)
            h = structure_to_heights(struct, cells) if struct is not None else INVALID
            if struct is not None:
                priors = (priors + [struct])[-10:]
        else:
            already = list(dict.fromkeys(emitted)) if arm == "schema_dedupe" else []
            p = prompt_a1(cells, top_view, already)
            text = ask(client, p, budget, thinking=thinking)
            h = parse_a1(text, len(cells))
        emitted.append(h)
        if h != INVALID:
            n_llm += 1

    pre = core.score(emitted, cells, gt, k)

    n_filled = 0
    if arm == "schema_dedupe":
        # exact enumeration as an auditor: fill only slots the model left
        # duplicated or unparsed.
        used = set(h for h in emitted if h != INVALID)
        pool = [g for g in gt if g not in used]
        for i, h in enumerate(list(emitted)):
            dup = h == INVALID or emitted[:i].count(h) > 0
            if dup and pool:
                emitted[i] = pool.pop(0)
                n_filled += 1

    sc = core.score(emitted, cells, gt, k)
    sc.update({"observation_set_id": sample["observation_set_id"], "n_blocks": sample["n_blocks"],
               "k": k, "n_llm": n_llm, "n_filled": n_filled,
               "V_pre": pre["V"], "U_pre": pre["U"], "R_pre": pre["R"]})
    return sc
