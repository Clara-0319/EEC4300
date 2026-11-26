"""The competing arms.  Everything that differs between them is in this file.

Each arm receives the same `Sample`, the same budget `k` (number of hypotheses it is
expected to deliver) and, if it wants one, the same LLM client.  It returns

    {"emitted": [graph, ...],       # what the arm claims, in order
     "n_calls": int,                # LLM calls (0 for symbolic arms)
     "search_states": int,          # DFS expansions (0 for LLM arms)
     "usage": {...} | None,         # token/cost accounting
     "info": {...}}                 # arm-specific diagnostics (repair stats, ...)

Arms
----
oracle          exact enumeration of the valid set, then coverage selection.
                Upper bound; tells you the ceiling, not a competing method.
symbolic        the same search under an explicit *state* budget (CCGIS-style).
                This is the honest "no LLM, no tokens" reference point.
random          the -LLM ablation for C3-Gen: identical repair + coverage pipeline,
                LLM replaced by uniform sampling inside the allowed edge set.
llm_baseline    a faithful re-implementation of the original benchmark loop:
                k sequential single-graph prompts, no repair, no slate selection.
c3gen           C3-Gen: one constraint-conditioned call asking for the full slate,
                minimal repair, coverage selection, one criticism round if short.
"""

from __future__ import annotations

import random
import re
import time
from textwrap import dedent as _dedent
from typing import Dict, List, Optional, Sequence, Tuple

import core
from core import Constraints, Graph, graph_key


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #
# Byte-for-byte copy of CausalBenchmarkLLM.create_prompt so the baseline arm is
# truly the published baseline; tests/test_prompts.py asserts the equality.
BASELINE_TEMPLATE = """
        You are given observations from perturbation experiments on a causal system.
        
        Semantics:
        - When a node is perturbed, the perturbed node is 0.
        - A node is 1 if it is a downstream descendant of the perturbed node in the causal graph.
        - All other nodes are 0.
        
        Nodes: {nodes_str}{constraint_info}
        
        Observations:
        {obs_block}
        
        Prior predictions (do not repeat if avoidable):
        {prior_block}
        
        Task:
        Output a single directed acyclic graph (DAG) over the nodes above that explains all observations.
        
        Diversity rule:
        - A "diverse" graph is any valid graph whose edge set is NOT identical to any prior prediction.
        - Generate diverse graphs when possible to explore the solution space.
        
        Formatting rules:
        1) Use only the listed nodes. No self-loops. No cycles.
        2) Respond with exactly one line:
        - If there are edges: Graph: A->B, B->C
        - If there are no edges: Graph: No edges
        """


def _dedent_strip(s: str) -> str:
    """Exactly what the paper does: textwrap.dedent then strip.

    It looks like it should remove the 8-space indent, but `constraint_info` starts
    with a newline, so the "Constraint:" line sits at column 0 and dedent() finds no
    common prefix -- the published prompt therefore *keeps* the indentation.  Matching
    this byte-for-byte is the point; a tidier prompt would be a different baseline.
    """
    return _dedent(s).strip()


def baseline_prompt(nodes, observations, prior: List[Graph], max_edges: Optional[int]) -> str:
    nodes_str = ", ".join(nodes)
    obs_block = "\n".join(o.get("string", "") for o in observations)
    if prior:
        lines = []
        for h in prior:
            edges = [f"{s}->{d}" for s, d in sorted(h)]
            lines.append("Graph: " + ", ".join(edges) if edges else "Graph: No edges")
        prior_block = "\n".join(lines)
    else:
        prior_block = "None"
    constraint_info = ""
    if max_edges is not None:
        constraint_info = f"\nConstraint: The graph should have at most {max_edges} edges."
    return _dedent_strip(BASELINE_TEMPLATE.format(
        nodes_str=nodes_str,
        constraint_info=constraint_info,
        obs_block=obs_block,
        prior_block=prior_block,
    ))


def c3_prompt(
    nodes,
    observations,
    c: Constraints,
    need: int,
    slate: List[Graph],
    criticism: Optional[str] = None,
    max_edges: Optional[int] = None,
) -> str:
    """Constraint-conditioned, slate-first prompt.

    Three differences from the baseline prompt, and each one is necessary:
    1. the observations are re-stated as compiled constraints (reach / must-not-reach),
       which removes the reasoning step the LLM was silently getting wrong;
    2. an explicit edge whitelist -- edges outside it are *provably* wrong, so there is
       no reason to let the model spend tokens on them;
    3. it asks for the whole slate in one call, which is what makes the token cost
       comparable to the baseline's k calls instead of k times larger.
    """
    lines = ["You are given perturbation experiments on a causal system.", ""]
    lines.append("Semantics: perturbing s sets s to 0; a node x is 1 if and only if x is a")
    lines.append("downstream descendant of s in the causal graph; every other node is 0.")
    lines.append("")
    lines.append("Nodes: " + ", ".join(nodes))
    if max_edges is not None:
        lines.append(f"Constraint: the graph has at most {max_edges} edges.")
    lines.append("")
    lines.append("Observations:")
    for o in observations:
        lines.append("  " + o.get("string", ""))
    lines.append("")
    lines.append("Hard constraints. These follow logically from the observations; any graph")
    lines.append("that violates one of them cannot explain the data:")
    for line in c.to_lines():
        lines.append("  " + line)
    lines.append("")
    lines.append("Edge whitelist. You may ONLY use the following directed edges. Every edge")
    lines.append("not listed here is logically impossible for this observation set:")
    lines.append("  " + ", ".join(f"{u}->{v}" for u, v in c.allowed))
    lines.append("")
    lines.append(f"Task: output exactly {need} distinct directed acyclic graphs. All {need} must")
    lines.append("differ from each other (no two may have the same edge set), and you should")
    lines.append("spread them out: prefer graphs that differ from the earlier ones by several")
    lines.append("edges, not by one.")
    if slate:
        lines.append("")
        lines.append("Graphs already accepted (do NOT repeat these, and stay far from them):")
        for i, g in enumerate(slate, 1):
            edges = [f"{u}->{v}" for u, v in sorted(g)]
            lines.append(f"  {i}. " + (", ".join(edges) if edges else "No edges"))
    if criticism:
        lines.append("")
        lines.append("Problems found in your previous answer, fix all of them:")
        lines.append(criticism)
    lines.append("")
    lines.append("Output format, one graph per line, exactly " + str(need) + " lines, nothing else:")
    lines.append("Graph 1: A->B, B->C")
    lines.append("Graph 2: No edges")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
_EDGE_RE = re.compile(r"([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)")


def _clean_line(line: str) -> str:
    line = line.replace("```", "").strip()
    line = (line.replace("\u2192", "->").replace("-->", "->").replace("=>", "->")
            .replace("\u2013>", "->").replace("->>", "->"))
    line = re.sub(r"^\s*(?:Graph\s*\d*\s*:|[-*]\s*)", "", line, flags=re.I)
    return line.strip().rstrip(".;")


def parse_single(response: str, nodes: Sequence[str], max_edges: Optional[int]) -> Optional[Graph]:
    """Same contract as the original parse_llm_response: one graph or None."""
    if not isinstance(response, str) or not response.strip():
        return None
    s = response.replace("```", "")
    m = re.search(r"(?i)\bgraph\s*:\s*(.+)$", s, flags=re.MULTILINE)
    line = m.group(1).strip() if m else s.strip().splitlines()[0].strip()
    line = _clean_line(line)
    if not line:
        return None
    if re.search(r"\b(no\s+edges?|empty|none|null)\b", line, re.I):
        return frozenset()
    edges = set()
    for u, v in _EDGE_RE.findall(line):
        if u not in nodes or v not in nodes or u == v:
            return None
        edges.add((u, v))
    if not edges:
        return None
    if max_edges is not None and len(edges) > max_edges:
        return None
    if not core.is_dag(nodes, edges):
        return None
    return frozenset(edges)


def parse_multi(response: str, nodes: Sequence[str], max_edges: Optional[int]) -> List[Graph]:
    """Parse a slate.  Malformed lines are dropped, not fatal: a bad line costs one
    slot, which the metrics then charge for, exactly like a failed baseline query."""
    if not isinstance(response, str):
        return []
    out: List[Graph] = []
    for raw in response.replace("```", "").splitlines():
        line = raw.strip()
        if not re.match(r"(?i)^\s*(graph\s*\d*|[-*]|\d+[.)])\s*:", line) and not re.match(
            r"(?i)^\s*graph\b", line
        ):
            continue
        body = _clean_line(line)
        if re.search(r"\b(no\s+edges?|empty)\b", body, re.I):
            out.append(frozenset())
            continue
        edges = set()
        bad = False
        for u, v in _EDGE_RE.findall(body):
            if u not in nodes or v not in nodes or u == v:
                bad = True
                break
            edges.add((u, v))
        if bad or not edges:
            continue
        if max_edges is not None and len(edges) > max_edges:
            continue
        if not core.is_dag(nodes, edges):
            continue
        out.append(frozenset(edges))
    return out


# --------------------------------------------------------------------------- #
# symbolic arms
# --------------------------------------------------------------------------- #
def run_oracle(sample, k, rng, llm=None, state_budget=None):
    t0 = time.perf_counter()
    c = sample["constraints"]
    cands = core.enumerate_all(c, limit=state_budget)
    slate = core.farthest_first(cands, k, seed=sample["seed"])
    return {
        "emitted": slate,
        "n_calls": 0,
        "search_states": len(cands),
        "usage": None,
        "info": {"n_candidates": len(cands), "wall": time.perf_counter() - t0},
    }


def run_symbolic(sample, k, rng, llm=None, state_budget=20_000):
    """CCGIS as it should have been: exact search capped by a *state* budget."""
    t0 = time.perf_counter()
    c = sample["constraints"]
    cands = core.enumerate_all(c, limit=state_budget)
    capped = len(cands) >= state_budget
    slate = core.farthest_first(cands, k, seed=sample["seed"])
    return {
        "emitted": slate,
        "n_calls": 0,
        "search_states": len(cands),
        "usage": None,
        "info": {
            "n_candidates": len(cands),
            "hit_state_cap": capped,
            "n_pruned_edges": c.n_pruned_edges(),
            "wall": time.perf_counter() - t0,
        },
    }


def run_random(sample, k, rng, llm=None, attempts_per_slot=60, pool_factor=3):
    """The -LLM ablation: no model, but the same repair + coverage pipeline.

    This arm exists to answer the only question that matters for the write-up:
    how much of C3-Gen's score comes from the LLM and how much from the symbolic
    scaffolding.  If this arm ties with c3gen, the LLM is decoration.

    It is deliberately *not* uniform rejection sampling over all DAGs: that would
    mostly measure the rejection rate.  It gets the same compiled constraints and the
    same repair step as C3-Gen, and only the generator is replaced by chance.  So the
    difference between it and c3gen is attributable to the model alone.
    """
    t0 = time.perf_counter()
    nodes = sample["nodes"]
    c = sample["constraints"]
    pool: List[Graph] = []
    for _ in range(k * pool_factor):
        if len(pool) >= k and len({graph_key(g) for g in pool}) >= k:
            break
        g = _random_valid(nodes, c, rng, attempts_per_slot)
        if g is not None:
            pool.append(g)
    slate = core.farthest_first(pool, k, seed=sample["seed"])
    return {
        "emitted": slate,
        "n_calls": 0,
        "search_states": 0,
        "usage": None,
        "info": {"n_candidates": len(pool),
                 "n_unique_candidates": len({graph_key(g) for g in pool}),
                 "wall": time.perf_counter() - t0},
    }


def _random_valid(nodes, c: Constraints, rng: random.Random, attempts: int) -> Optional[Graph]:
    """A random graph drawn from the valid set, by construction rather than by search.

    1. random topological order consistent with the required-reachability closure
       (so a required s -> x edge is always allowed to point forward);
    2. one random forward path per (s, x) requirement;
    3. random extra forward whitelisted edges.
    No backtracking, no enumeration: this is what "no model" looks like.
    """
    for _ in range(attempts):
        order = _required_topological_order(nodes, c, rng)
        if order is None:
            continue
        pos = {n: i for i, n in enumerate(order)}
        fwd = [(u, v) for (u, v) in c.allowed if pos[u] < pos[v]]
        fwd_adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for u, v in fwd:
            fwd_adj[u].append(v)
        edges: set = set()
        ok = True
        for s in sorted(c.required):
            for x in sorted(c.required[s]):
                if x in edges_reach(edges, nodes, s):
                    continue
                path = _random_forward_path(fwd_adj, rng, s, x)
                if path is None:
                    ok = False
                    break
                for a, b in zip(path, path[1:]):
                    edges.add((a, b))
            if not ok:
                break
        if not ok:
            continue
        p = rng.uniform(0.15, 0.55)
        for e in fwd:
            if rng.random() < p:
                edges.add(e)
        g = frozenset(edges)
        if core.is_dag(nodes, g) and core.validate(nodes, g, c):
            return g
    return None


def edges_reach(edges, nodes, s):
    adj = core.adjacency(nodes, edges)
    return core.reach_from(adj, s)


def _required_topological_order(nodes, c: Constraints, rng: random.Random):
    """Kahn with random tie-breaking over the required-reachability closure."""
    adj: Dict[str, set] = {n: set() for n in nodes}
    indeg = {n: 0 for n in nodes}
    for s, need in c.required.items():
        for x in sorted(need):
            if x not in adj[s]:
                adj[s].add(x)
                indeg[x] += 1
    # transitive closure so that a path a->b->c also orders a before c
    changed = True
    while changed:
        changed = False
        for a in nodes:
            for b in sorted(adj[a]):
                for cc in sorted(adj[b]):
                    if cc not in adj[a] and cc != a:
                        adj[a].add(cc)
                        changed = True
    # indeg is now wrong for closure edges; rebuild
    indeg = {n: 0 for n in nodes}
    for a in nodes:
        for b in sorted(adj[a]):
            indeg[b] += 1
    ready = [n for n in nodes if indeg[n] == 0]
    out = []
    while ready:
        i = rng.randrange(len(ready))
        n = ready.pop(i)
        out.append(n)
        for m in sorted(adj[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    if len(out) != len(nodes):
        return None  # required constraints are cyclic -> no valid graph exists
    return out


def _random_forward_path(adj: Dict[str, List[str]], rng, s, x):
    """Shortest-ish random forward path from s to x (BFS with shuffled neighbours)."""
    if s == x:
        return [s]
    prev = {s: None}
    queue = [s]
    while queue:
        u = queue.pop(0)
        nbrs = list(adj.get(u, []))
        rng.shuffle(nbrs)
        for v in nbrs:
            if v not in prev:
                prev[v] = u
                if v == x:
                    path = [x]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                queue.append(v)
    return None


# --------------------------------------------------------------------------- #
# LLM arms
# --------------------------------------------------------------------------- #
def run_llm_baseline(sample, k, rng, llm, max_retries=1):
    """Original loop: k sequential calls, one graph each, prior graphs in the prompt."""
    t0 = time.perf_counter()
    nodes = sample["nodes"]
    c = sample["constraints"]
    obs = sample["observations"]
    max_edges = sample.get("max_edges")
    emitted: List[Graph] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost = 0.0
    n_calls = 0
    for _ in range(k):
        prompt = baseline_prompt(nodes, obs, emitted, max_edges)
        g = None
        for _attempt in range(max_retries):
            res = llm.query_with_usage(prompt)
            n_calls += 1
            u = res.get("usage") or {}
            for key in usage:
                usage[key] += u.get(key, 0) or 0
            cost += res.get("cost", 0.0) or 0.0
            g = parse_single(res.get("response", ""), nodes, max_edges)
            if g is not None:
                break
        if g is not None:
            emitted.append(g)
    return {
        "emitted": emitted,
        "n_calls": n_calls,
        "search_states": 0,
        "usage": usage,
        "cost": cost,
        "info": {"wall": time.perf_counter() - t0},
    }


def run_c3gen_hybrid(sample, k, rng, llm, max_rounds=2):
    """C3-Gen, then fill the slots the LLM left empty by exact constrained search.

    The LLM is a good *proposer* (it rarely breaks the observations) and a bad
    *enumerator* (it cannot list 20-50 distinct compatible DAGs).  Constrained search is
    the opposite.  Filling the gap with enumeration costs milliseconds and makes the arm
    complete up to the cap, so the provenance split (`n_llm` vs `n_filled`) is the
    interesting quantity: it says how much of the slate the model actually earned.
    """
    res = run_c3gen(sample, k, rng, llm, max_rounds=max_rounds)
    slate = list(res["emitted"])
    n_llm = len(slate)
    n_filled = 0
    if len(slate) < k:
        c = sample["constraints"]
        have = {graph_key(g) for g in slate}
        rest = [g for g in core.enumerate_all(c) if graph_key(g) not in have]
        picked = core.farthest_first(rest, k - len(slate), seed=sample["seed"])
        slate.extend(picked)
        n_filled = len(picked)
    slate = core.farthest_first(slate, k, seed=sample["seed"])
    info = dict(res.get("info") or {})
    info.update({"n_llm": n_llm, "n_filled": n_filled})
    return {
        "emitted": slate,
        "n_calls": res["n_calls"],
        "search_states": res["search_states"],
        "usage": res["usage"],
        "cost": res["cost"],
        "info": info,
    }


def run_c3gen(sample, k, rng, llm, max_rounds=2):
    """C3-Gen: constraint-conditioned slate in one call, repair, coverage, criticism."""
    t0 = time.perf_counter()
    nodes = sample["nodes"]
    c = sample["constraints"]
    obs = sample["observations"]
    max_edges = sample.get("max_edges")
    slate: List[Graph] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost = 0.0
    n_calls = 0
    repair_stats = {"removed": 0, "added": 0, "unrepairable": 0}
    rounds_info = []
    criticism = None

    for _round in range(max_rounds):
        need = k - len(slate)
        if need <= 0:
            break
        prompt = c3_prompt(nodes, obs, c, need, slate, criticism, max_edges)
        res = llm.query_with_usage(prompt)
        n_calls += 1
        u = res.get("usage") or {}
        for key in usage:
            usage[key] += u.get(key, 0) or 0
        cost += res.get("cost", 0.0) or 0.0
        raw = parse_multi(res.get("response", ""), nodes, max_edges)

        kept = 0
        for g in raw:
            if core.validate(nodes, g, c):
                fixed = g
            else:
                fixed, st = core.repair(nodes, g, c)
                for key in repair_stats:
                    repair_stats[key] += st[key]
                if not fixed or not core.validate(nodes, fixed, c):
                    continue
            if graph_key(fixed) not in {graph_key(x) for x in slate}:
                slate.append(fixed)
                kept += 1
            if len(slate) >= k:
                break

        rounds_info.append({"parsed": len(raw), "kept": kept, "slate": len(slate)})

        if len(slate) >= k:
            break
        criticism = _criticise(raw, c, slate, nodes) if raw else (
            "The previous answer contained no parsable graph line. "
            "Use exactly the format: Graph 1: A->B, B->C"
        )

    slate = core.farthest_first(slate, k, seed=sample["seed"])
    return {
        "emitted": slate,
        "n_calls": n_calls,
        "search_states": 0,
        "usage": usage,
        "cost": cost,
        "info": {"repair": repair_stats, "rounds": rounds_info,
                 "wall": time.perf_counter() - t0},
    }


def _criticise(raw: List[Graph], c: Constraints, slate: List[Graph], nodes) -> str:
    """Structural criticism: name the violated constraint, not "try again"."""
    lines: List[str] = []
    bad = [g for g in raw if not core.validate(nodes, g, c)]
    if bad:
        for g in bad[:3]:
            illegal = [f"{u}->{v}" for (u, v) in sorted(g) if (u, v) not in c.allowed_set]
            if illegal:
                lines.append("- these edges are not on the whitelist: " + ", ".join(illegal))
            else:
                adj = core.adjacency(nodes, list(g))
                for s, need in sorted(c.required.items()):
                    miss = sorted(need - core.reach_from(adj, s))
                    if miss:
                        lines.append(f"- in one graph {s} does not reach {', '.join(miss)}, "
                                     "but the observation requires it")
                        break
    if len(slate) < 2:
        lines.append("- you returned too few distinct graphs; every line must be a different DAG")
    if not lines:
        lines.append("- the graphs repeated each other; each line must differ by several edges")
    return "\n".join(lines)
