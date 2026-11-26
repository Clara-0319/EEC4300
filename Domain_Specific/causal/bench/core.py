"""Core symbolic machinery for the causal hypothesis-generation benchmark.

Why this module exists
----------------------
The old pipeline (`run_causal_specialized.py`) duplicated the validation predicate
in three places and silently changed the *denominator* of every metric, which made
CCGIS look like it scored 100% when it was only ever measured on the graphs it had
already guaranteed.  Everything that decides "is this graph valid" and "what is this
score" now lives here, once, so every arm (LLM baseline, C3-Gen, random, oracle) is
scored by identical code.

Graph representation
--------------------
A graph is a `frozenset` of (u, v) string pairs over a fixed node list.
Canonical identity is `graph_key(g) = tuple(sorted(g))` -- no md5, no CausalGraph
object, so graphs are hashable, cheap to compare and cheap to store.

Key theoretical fact used below (proved immediately after this block)
-----------------------------------------------------------
For an observation (s, effects):  x is 1  <=>  x is a *descendant* of s.
So with R_s = {x : effects[x] == 1} and F_s = {x != s : effects[x] == 0}:

    edge (u, v) may appear in a valid graph
        <=>  there is NO observation s with  u in R_s + {s}  and  v in F_s

Proof: such an edge would make v reachable from s, contradicting F_s.  Conversely,
any violation of a forbidden set requires a path s ->...-> u -> v with u reachable,
i.e. exactly such an edge.  A valid graph is therefore *exactly* one that
  (a) uses only allowed edges, (b) is acyclic, (c) reaches every x in R_s from s.
Condition (a)+(b) can never violate a forbidden set, so validation reduces to (c).
That single theorem is what makes the search output-sensitive.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

Graph = FrozenSet[Tuple[str, str]]


# --------------------------------------------------------------------------- #
# basic graph primitives (no networkx: these are called millions of times)
# --------------------------------------------------------------------------- #
def adjacency(nodes: Sequence[str], edges: Iterable[Tuple[str, str]]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
    return adj


def reach_from(adj: Dict[str, List[str]], s: str) -> Set[str]:
    """Nodes reachable from s, excluding s itself."""
    seen = {s}
    stack = [s]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    seen.discard(s)
    return seen


def is_dag(nodes: Sequence[str], edges: Iterable[Tuple[str, str]]) -> bool:
    """Kahn's algorithm; also rejects self loops."""
    indeg = {n: 0 for n in nodes}
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        if u == v:
            return False
        adj[u].append(v)
        indeg[v] += 1
    queue = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while queue:
        u = queue.pop()
        seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return seen == len(nodes)


def graph_key(g: Iterable[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(g))


def topo_orders(nodes: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    return itertools.permutations(nodes)


# --------------------------------------------------------------------------- #
# constraint compilation
# --------------------------------------------------------------------------- #
class Constraints:
    """The compiled, logically equivalent restatement of an observation set."""

    __slots__ = ("nodes", "required", "forbidden", "allowed", "allowed_set", "obs")

    def __init__(self, nodes, required, forbidden, allowed, obs):
        self.nodes = list(nodes)
        self.required: Dict[str, Set[str]] = required      # s -> must be reachable
        self.forbidden: Dict[str, Set[str]] = forbidden    # s -> must not be reachable
        self.allowed: List[Tuple[str, str]] = allowed      # every edge a valid graph may use
        self.allowed_set = set(allowed)
        self.obs = obs

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_observations(cls, nodes: Sequence[str], observations: List[Dict]) -> "Constraints":
        required: Dict[str, Set[str]] = {}
        forbidden: Dict[str, Set[str]] = {}
        for o in observations:
            s = o["perturbed_node"]
            eff = o["effects"]
            required[s] = {x for x, v in eff.items() if v == 1 and x != s}
            forbidden[s] = {x for x, v in eff.items() if v == 0 and x != s}
            if eff.get(s, 0) != 0:  # defence: the dataset guarantees 0
                raise ValueError(f"observation on {s} claims {s}={eff.get(s)}")

        allowed: List[Tuple[str, str]] = []
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                ok = True
                for s in required:
                    if u in required[s] or u == s:
                        if v in forbidden[s]:
                            ok = False
                            break
                if ok:
                    allowed.append((u, v))
        return cls(nodes, required, forbidden, allowed, observations)

    # -- derived quantities ------------------------------------------------ #
    def n_pruned_edges(self) -> int:
        n = len(self.nodes)
        return n * (n - 1) - len(self.allowed)

    def to_lines(self) -> List[str]:
        """Human/LLM readable form of the compiled constraints."""
        out = []
        for s in sorted(self.required):
            req = sorted(self.required[s])
            forb = sorted(self.forbidden[s])
            parts = [f"Perturb({s}):"]
            parts.append(f"must reach {', '.join(req) if req else '(nothing)'}")
            parts.append(f"; must NOT reach {', '.join(forb) if forb else '(nothing)'}")
            out.append(" ".join(parts))
        return out


def validate(nodes: Sequence[str], g: Iterable[Tuple[str, str]], c: Constraints) -> bool:
    """Exact validity check, equivalent to the dataset's effect comparison."""
    edges = list(g)
    if any(e not in c.allowed_set for e in edges):
        return False
    if not is_dag(nodes, edges):
        return False
    adj = adjacency(nodes, edges)
    for s, need in sorted(c.required.items()):
        if not need.issubset(reach_from(adj, s)):
            return False
    return True


# --------------------------------------------------------------------------- #
# exact enumeration (the oracle / "unlimited budget" upper bound)
# --------------------------------------------------------------------------- #
def enumerate_all(c: Constraints, limit: Optional[int] = None) -> List[Graph]:
    """Every valid graph, by DFS over topological orders of the allowed subgraph.

    Each valid graph is produced once (results are deduplicated by graph_key).
    A DAG is *any* edge subset of the forward pairs of some topological order, so
    iterating orders and then forward subsets is complete.
    """
    nodes = c.nodes
    allowed = c.allowed_set
    required = c.required
    found: Dict[Tuple[Tuple[str, str], ...], Graph] = {}

    # per observation: requirement list, used for the feasibility prune
    reqs = [(s, frozenset(need)) for s, need in required.items() if need]

    for order in topo_orders(nodes):
        pos = {n: i for i, n in enumerate(order)}
        cand = [(u, v) for (u, v) in c.allowed if pos[u] < pos[v]]

        def feasible(chosen: Set[Tuple[str, str]], start: int) -> bool:
            """Can every still-unsatisfied requirement be met with chosen + cand[start:]?"""
            if not reqs:
                return True
            edges = list(chosen) + cand[start:]
            adj = adjacency(nodes, edges)
            for s, need in reqs:
                got = reach_from(adj, s)
                for x in sorted(need):
                    if x not in got:
                        return False
            return True

        if not feasible(set(), 0):
            continue

        stack: List[Tuple[int, Set[Tuple[str, str]]]] = [(0, set())]
        while stack:
            i, chosen = stack.pop()
            # emit if valid (an already-valid set stays valid when edges are added,
            # because allowed edges never reach a forbidden node)
            adj = adjacency(nodes, chosen)
            if all(need.issubset(reach_from(adj, s)) for s, need in reqs):
                k = graph_key(chosen)
                if k not in found:
                    found[k] = frozenset(chosen)
                    if limit and len(found) >= limit:
                        return list(found.values())
            for j in range(i, len(cand)):
                nxt = chosen | {cand[j]}
                if feasible(nxt, j + 1):
                    stack.append((j + 1, nxt))
    return list(found.values())


def count_valid(c: Constraints, limit: int = 200_000) -> int:
    return len(enumerate_all(c, limit=limit))


# --------------------------------------------------------------------------- #
# repair: turn an almost-valid graph into a valid one with minimal edits
# --------------------------------------------------------------------------- #
def repair(nodes: Sequence[str], g: Iterable[Tuple[str, str]], c: Constraints) -> Tuple[Graph, Dict]:
    """Minimal add/remove-edge repair.

    Step 1 drops every edge outside the whitelist (those are the forbidden-reachability
    violations; each dropped edge is one violated constraint unit).
    Step 2 adds the shortest allowed path for every still-unreached required node.
    """
    edges = set(g)
    stats = {"removed": 0, "added": 0, "unrepairable": 0}

    for e in list(edges):
        if e not in c.allowed_set:
            edges.discard(e)
            stats["removed"] += 1

    for s, need in sorted(c.required.items()):
        for x in sorted(need):
            adj = adjacency(nodes, edges)
            if x in reach_from(adj, s):
                continue
            path = _shortest_allowed_path(nodes, edges, c, s, x)
            if path is None:
                stats["unrepairable"] += 1
                continue
            for e in path:
                if e not in edges:
                    edges.add(e)
                    stats["added"] += 1

    if not is_dag(nodes, edges):
        # extremely rare: a repair closed a cycle -> fall back to a valid subgraph
        stats["unrepairable"] += 1
        return frozenset(), stats
    return frozenset(edges), stats


def _shortest_allowed_path(nodes, edges, c, s, x) -> Optional[List[Tuple[str, str]]]:
    """BFS over allowed edges not yet used, avoiding cycles with the current graph."""
    adj = adjacency(nodes, edges)
    blocked = {s}
    # nodes already downstream of s cannot receive an edge back (would cycle)
    blocked |= reach_from(adj, s)
    parent: Dict[str, Tuple[str, str]] = {}
    queue = [s]
    seen = {s}
    while queue:
        u = queue.pop(0)
        for (a, b) in c.allowed:
            if a != u or b in seen:
                continue
            if b in blocked:
                continue
            if (a, b) in edges:
                seen.add(b)
                parent[b] = (a, b)
                queue.append(b)
                continue
            seen.add(b)
            parent[b] = (a, b)
            queue.append(b)
            if b == x:
                path = []
                cur = x
                while cur != s:
                    e = parent.get(cur)
                    if e is None:
                        return None
                    path.append(e)
                    cur = e[0]
                return list(reversed(path))
    return None


# --------------------------------------------------------------------------- #
# selection: which k graphs to report
# --------------------------------------------------------------------------- #
def farthest_first(candidates: List[Graph], k: int, seed: int = 0) -> List[Graph]:
    """Pick k graphs that are maximally different from each other (edge-set distance).

    Deterministic given (candidates, k, seed): ties broken by graph_key so results are
    reproducible across runs, and the same slate is produced for every arm.
    """
    uniq: Dict[Tuple[Tuple[str, str], ...], Graph] = {}
    for g in candidates:
        uniq.setdefault(graph_key(g), g)
    pool = [uniq[key] for key in sorted(uniq)]
    if len(pool) <= k:
        return pool

    rng = random.Random(seed)
    chosen = [pool[rng.randrange(len(pool))]]
    remaining = [g for g in pool if g is not chosen[0]]
    while len(chosen) < k and remaining:
        best_idx, best_d = -1, -1
        for idx, g in enumerate(remaining):
            d = min(_sym_diff(g, h) for h in chosen)
            if d > best_d:
                best_d, best_idx = d, idx
        chosen.append(remaining.pop(best_idx))
    return chosen


def _sym_diff(a: Graph, b: Graph) -> int:
    return len(a ^ b)


# --------------------------------------------------------------------------- #
# metrics -- one definition, used by every arm
# --------------------------------------------------------------------------- #
def score(
    emitted: Sequence[Graph],
    nodes: Sequence[str],
    c: Constraints,
    gt: Sequence[Graph],
    k: int,
) -> Dict[str, float]:
    """All metrics for one sample.

    k is the *requested output budget* (identical across arms), not len(emitted).
    Using len(emitted) as the denominator is exactly the bug that produced the
    "42% -> 100%" drift; it is kept here only as an auxiliary pool-convention view.
    """
    gt_keys = {graph_key(g) for g in gt}
    valid = [g for g in emitted if validate(nodes, g, c)]
    seen: Dict[Tuple[Tuple[str, str], ...], Graph] = {}
    for g in valid:
        seen.setdefault(graph_key(g), g)
    uniq_valid = list(seen.values())

    n_gt = len(gt_keys)
    recovered = {graph_key(g) for g in uniq_valid} & gt_keys
    cap = min(k, n_gt) / n_gt if n_gt else 0.0
    rec = len(recovered) / n_gt if n_gt else 0.0

    n_em = len(emitted)
    return {
        "n_gt": n_gt,
        "n_emitted": n_em,
        "n_valid": len(valid),
        "n_unique_valid": len(uniq_valid),
        "n_recovered": len(recovered),
        # strict convention (baseline): denominator = requested budget
        "V": len(valid) / k if k else 0.0,
        "U": (len(uniq_valid) / k) if k else 0.0,
        "R": rec,
        "cap": cap,
        # pool convention (the one CCGIS was accidentally scored with)
        "V_pool": len(valid) / n_em if n_em else 0.0,
        "U_pool": (len(uniq_valid) / len(valid)) if valid else 0.0,
        # decomposition of V: did it use every slot, and was every claim valid?
        "purity": (len(valid) / n_em) if n_em else 0.0,
        "fill": n_em / k if k else 0.0,
        # the only number that is comparable across different |GT|
        "R_norm": (rec / cap) if cap > 0 else 0.0,
        "slot_fill": n_em / k if k else 0.0,
    }
