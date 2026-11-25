"""Shared deterministic task adapters; no model calls or reference answers in prompts."""
from functools import lru_cache
from itertools import product
import json
import networkx as nx

from boolean_control_protocol import decode, complexity
from settings import MECHANISTIC
from compositional.upstream import BooleanAdapter, VoxelAdapter, load


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def upstream_error_text(record_error):
    """Render a transport error in the upstream 'Error querying ...' style."""
    kind = str(record_error.get('type') or 'unknown')
    text = 'Error querying LLM: ' + kind
    status = record_error.get('status')
    if status is not None:
        text += f' HTTP Error {status}'
    return text


def upstream_error_detail(adapter, max_retries, error_text):
    """Boolean-style prior-history entry appended after a fully failed upstream round."""
    return f'[ERROR after {max_retries} attempts] Type: {adapter.classify_error(error_text)} | {error_text}'


def upstream_history_add(adapter, case, history, seen, payload, error_detail=None):
    """Replicate upstream prior-history maintenance; shared by runs and audit replay."""
    if payload is not None:
        if case['task'] == '3d':
            key = adapter.upstream_hash(payload)
            if key in seen:
                return
            seen.add(key)
        history.append(payload)
    elif error_detail is not None and case['task'] == 'boolean':
        history.append(error_detail)


def rejected(reason='parse_or_space_error'):
    return {'in_space': False, 'valid': False, 'key': None, 'complexity': None, 'reason': reason}


@lru_cache(None)
def bool_catalog(level):
    ops = ('AND', 'OR', 'NOT') if level == 'medium' else ('AND', 'OR', 'NOT', 'NOR')
    data = load('boolean/boolean_dataset.py')
    pool = data.BooleanDiscoveryGame.generate_all_expressions(['x', 'y'], set(ops), 2, MECHANISTIC)
    result = {}
    for e in pool:
        key, truth = decode(e.formula)
        assert key == e.mechanistic_key(**MECHANISTIC) and truth == e.truth_table
        result[repr(key)] = {'payload': e.formula, 'key': repr(key), 'complexity': complexity(key),
                            'truth_table': {str(k): v for k, v in truth.items()}}
    assert len(result) == (34 if level == 'medium' else 65)
    return result


def bool_metadata(level):
    return {'variables': ['x', 'y'], 'operators': ['AND', 'OR', 'NOT'] + (['NOR'] if level == 'hard' else []),
            'max_depth': 2, 'mechanistic_opts': MECHANISTIC}


class BooleanTask:
    def __init__(self, case):
        self.case = case
        self.bridge = BooleanAdapter(bool_metadata(case['difficulty']), case['observations'])
        self.catalog = bool_catalog(case['difficulty'])
        self.bridge.runner.operators = sorted(self.bridge.runner.operators)
        self.rules = ('Shared bounded language: primitive-gate depth <=2, no constants. '
            'Equivalent spellings of a member of this canonical bounded language are accepted. '
            'For complexity count canonical operator nodes; flattened AND/OR counts once and NOT(OR(...)) counts two. ')
        self.rules += ('NOR is an allowed primitive, one gate for depth (two nodes for complexity).' if case['difficulty'] == 'hard'
                       else 'NOR is not a primitive in this level; NOT applied to OR is allowed and uses two depth levels.')

    def parse(self, response):
        return self.bridge.parse(response)

    def evaluate(self, payload):
        try:
            key, truth = decode(payload)
            if repr(key) not in self.catalog:
                return rejected()
            valid = all(truth[(o['inputs']['x'], o['inputs']['y'])] == o['output'] for o in self.case['observations'])
            return {'in_space': True, 'valid': valid, 'key': repr(key), 'complexity': complexity(key),
                    'reason': 'valid' if valid else 'observation_mismatch'}
        except (ValueError, TypeError, KeyError, RecursionError):
            return rejected()

    def prompt(self, history):
        return self.bridge.prompt(history) + '\n' + self.rules

    def upstream_prompt(self, history):
        """Verbatim upstream create_prompt output; used by the replicated baseline."""
        return self.bridge.prompt(history)

    def classify_error(self, error_text):
        return self.bridge.runner._classify_error(error_text)


def voxel_key(payload):
    value = [[list(row) for row in layer] for layer in payload]
    while len(value) > 1 and not any(cell for row in value[-1] for cell in row):
        value.pop()
    return compact(value)


class VoxelTask:
    def __init__(self, case):
        self.case = case
        self.bridge = VoxelAdapter({'grid_size': 3, 'max_height': 3}, case['observation'])

    def parse(self, response):
        return self.bridge.parse(response)

    def evaluate(self, payload):
        verdict = self.bridge.evaluate(payload)
        if not verdict['in_space']:
            return rejected()
        # Full tensors, rather than shortened hashes, identify distinct hypotheses.
        return {**verdict, 'key': voxel_key(payload),
                'complexity': sum(cell for layer in payload for row in layer for cell in row)}

    def prompt(self, history):
        return self.bridge.prompt(history) + '\nReturn only one complete Structure in the required format. Complexity is total occupied voxels.'

    def upstream_prompt(self, history):
        """Verbatim upstream create_prompt output; used by the replicated baseline."""
        return self.bridge.prompt(history)

    def upstream_hash(self, payload):
        return self.bridge.data.Structure3D(payload).get_hash()

    def classify_error(self, error_text):
        return self.bridge.runner._classify_error(error_text)


def graph_key(edges):
    return compact(sorted(set(tuple(edge) for edge in edges)))


class CausalTask:
    def __init__(self, case):
        self.case = case
        self.module = load('causal/run_causal_benchmark.py')
        self.runner = self.module.CausalBenchmarkEnhanced.__new__(self.module.CausalBenchmarkEnhanced)
        self.runner.nodes = case['nodes']
        self.runner.max_edges = len(case['nodes']) * (len(case['nodes']) - 1) // 2

    def parse(self, response):
        obj = self.runner.parse_llm_response(response)
        return obj.to_dict() if obj is not None else None

    def evaluate(self, payload):
        try:
            if not isinstance(payload, dict) or sorted(payload['nodes']) != sorted(self.case['nodes']):
                return rejected()
            nodes = self.case['nodes']
            edges = payload['edges']
            if any(len(edge) != 2 or edge[0] == edge[1] or any(v not in nodes for v in edge) for edge in edges):
                return rejected()
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)
            if not nx.is_directed_acyclic_graph(graph):
                return rejected('cycle')
            valid = all({v: int(v in nx.descendants(graph, o['perturbed_node'])) for v in nodes} == o['effects']
                        for o in self.case['observations'])
            return {'in_space': True, 'valid': valid, 'key': graph_key(edges),
                    'complexity': graph.number_of_edges(), 'reason': 'valid' if valid else 'observation_mismatch'}
        except (KeyError, TypeError, ValueError, nx.NetworkXError):
            return rejected()

    def prompt(self, history):
        objects = [self.module.CausalGraph.from_dict(p) for p in history]
        return self.runner.create_prompt(self.case['observations'], objects) + '\nComplexity is the number of distinct directed edges.'

    def upstream_prompt(self, history):
        """Verbatim upstream create_prompt output; used by the replicated baseline."""
        objects = [self.module.CausalGraph.from_dict(p) for p in history]
        return self.runner.create_prompt(self.case['observations'], objects)

    def classify_error(self, error_text):
        return self.runner._classify_error(error_text)


def adapter_for(case):
    return {'boolean': BooleanTask, '3d': VoxelTask, 'causal': CausalTask}[case['task']](case)


def make_library(task, difficulty):
    if task == 'boolean':
        ops = bool_metadata(difficulty)['operators']
        expressions = ['x', 'y', 'NOT (x)', 'NOT (y)']
        for op in ops:
            if op == 'NOT':
                continue
            for a, b in product(('x', 'y'), repeat=2):
                expressions.append(f'NOR({a},{b})' if op == 'NOR' else f'({a} {op} {b})')
        seen, blocks = set(), []
        for expression in expressions:
            key, _ = decode(expression)
            if key not in seen:
                seen.add(key)
                blocks.append(expression)
        text = '\nFixed pre-generated depth<=1 building blocks:\n' + '\n'.join(f'P{i}: {s}' for i, s in enumerate(blocks))
        text += '\nUse a block directly or combine blocks with one outer allowed gate. A and B may be the same block. '
        text += 'Templates: A; NOT(A); (A AND B); (A OR B)' + ('; NOR(A,B)' if 'NOR' in ops else '')
        text += '. Generate ONE complete expression matching all observations, avoiding prior equivalents. Expand blocks into variables and operators; do not output IDs or intermediate choices.'
    elif task == '3d':
        blocks = [format(i, '09b') for i in range(512)]
        text = '\nFixed pre-generated library of ALL 512 possible 3x3 single-layer masks (row-major, 9 binary cells per mask):\n'
        text += '\n'.join(' '.join(blocks[i:i+16]) for i in range(0, 512, 16))
        text += '\nCompose ONE complete structure from 1 to 3 library layers, bottom to top. The bitwise OR must equal the observed top view; every upper 1 must be supported by a lower 1. '
        text += 'Output the full Structure with each layer expanded as three rows, not mask IDs or a partial choice. Avoid previous identical structures.'
    else:
        nodes = list('ABCDEF'[:5 if difficulty == 'medium' else 6])
        blocks = ['EMPTY'] + [f'{a}->{b}' for a in nodes for b in nodes if a != b]
        text = '\nFixed pre-generated graph-primitive library (all possible labeled single directed edges, plus empty):\n' + ', '.join(blocks)
        text += '\nCompose ONE complete DAG by taking a subset of these edge primitives (or EMPTY alone), merging shared node labels and removing duplicate edges. '
        text += 'Keep all listed nodes, introduce no cycles, satisfy all perturbation observations, and avoid prior identical edge sets. Output the full Graph edge list, not primitive indices or partial choices.'
    return {'task': task, 'difficulty': difficulty, 'blocks': blocks, 'count': len(blocks), 'prompt': text,
            'uses_observations': False, 'uses_ground_truth': False, 'api_calls_to_generate': 0}


def public_case(case):
    return {k: v for k, v in case.items() if k != 'references'}
