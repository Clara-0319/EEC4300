"""Frozen case construction and independent, exhaustive reference checks."""
from collections import Counter
from itertools import combinations, product
import random
import networkx as nx
import numpy as np

from matrix_adapters import bool_catalog, adapter_for, make_library, graph_key, decode, voxel_key
from compositional.upstream import load


def closure(n, edges):
    rows = [0] * n
    for a, b in edges:
        rows[a] |= 1 << b
    for k in range(n):
        for i in range(n):
            if rows[i] & (1 << k):
                rows[i] |= rows[k]
    return tuple(rows)


def causal_case(n, optional_count, seed):
    rng = random.Random(seed)
    nodes = list('ABCDEF'[:n])
    pairs = list(combinations(range(n), 2))
    for attempt in range(50000):
        edges = [edge for edge in pairs if rng.random() < 0.5]
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(edges)
        if not nx.is_weakly_connected(graph):
            continue
        transitive = nx.transitive_closure_dag(graph)
        reduction = nx.transitive_reduction(graph)
        required = sorted(reduction.edges())
        optional = sorted(set(transitive.edges()) - set(required))
        if len(optional) == optional_count:
            break
    else:
        raise ValueError('No matching causal instance')
    # Random relabeling hides the construction's topological order from the model.
    labels = nodes.copy()
    rng.shuffle(labels)
    references = []
    for bits in product((0, 1), repeat=len(optional)):
        edges = required + [e for b, e in zip(bits, optional) if b]
        edges = sorted([[labels[a], labels[b]] for a, b in edges])
        references.append({'payload': {'nodes': nodes, 'edges': edges}, 'key': graph_key(edges), 'complexity': len(edges)})
    observations = []
    for index, name in enumerate(labels):
        effects = {labels[v]: int(v in nx.descendants(transitive, index)) for v in range(n)}
        observations.append({'perturbed_node': name, 'effects': effects,
                             'string': f"Perturb({name}) -> " + ' '.join(f'{v}:{effects[v]}' for v in nodes)})
    observations.sort(key=lambda o: o['perturbed_node'])
    # Independent brute force over every subset of the full observed closure.
    # Edges outside it cannot match a fully observed descendant relation.
    all_edges = sorted(transitive.edges())
    expected = closure(n, all_edges)
    brute = set()
    for mask in range(1 << len(all_edges)):
        candidate = [e for i, e in enumerate(all_edges) if mask & (1 << i)]
        if closure(n, candidate) == expected:
            brute.add(graph_key([[labels[a], labels[b]] for a, b in candidate]))
    assert brute == {r['key'] for r in references}
    return {'task': 'causal', 'difficulty': 'medium' if n == 5 else 'hard',
            'case_id': f'causal_n{n}_M{len(references)}', 'nodes': nodes,
            'observations': observations, 'references': references,
            'construction': {'seed': seed, 'attempt': attempt, 'node_count': n,
                             'interventions': n, 'all_nodes_observed': True,
                             'target_optional_edges': optional_count, 'no_edge_cap_beyond_DAG': True,
                             'reference_completeness': 'required transitive reduction + arbitrary subset of transitive shortcuts; independently brute-force verified'}}


def generate():
    cases, libraries = [], {}
    for difficulty in ('medium', 'hard'):
        for task in ('boolean', '3d', 'causal'):
            libraries[f'{task}_{difficulty}'] = make_library(task, difficulty)
        catalog = bool_catalog(difficulty)
        for x, y, z in product((0, 1), repeat=3):
            refs = [r for r in catalog.values() if r['truth_table'][str((x, y))] == z]
            cases.append({'task': 'boolean', 'difficulty': difficulty,
                          'case_id': f'boolean_{difficulty}_{x}{y}_{z}',
                          'observations': [{'inputs': {'x': x, 'y': y}, 'output': z}],
                          'references': refs, 'construction': {'depth': 2, 'operator_set': 'extended' if difficulty == 'medium' else 'full',
                                                           'selection': 'all eight single-observation cases'}})
        # Verify fixed blocks compose the full bounded Boolean language.
        blocks = libraries[f'boolean_{difficulty}']['blocks']
        expressions = blocks + [f'NOT ({b})' for b in blocks]
        for left, right in product(blocks, repeat=2):
            expressions += [f'(({left}) AND ({right}))', f'(({left}) OR ({right}))']
            if difficulty == 'hard':
                expressions.append(f'NOR(({left}),({right}))')
        keys = set()
        for expr in expressions:
            try:
                keys.add(repr(decode(expr)[0]))
            except ValueError:
                pass
        assert keys == set(catalog)
        m = 2 if difficulty == 'medium' else 3
        selected = random.Random(33550336).sample(list(combinations(range(9), m)), 3)
        generator = load('3d/generate_3d_dataset_complete.py')
        for coords in selected:
            top = ''.join('1' if i in coords else '0' for i in range(9))
            structures = generator.enumerate_all_structures_by_heights(np.array(list(map(int, top))).reshape(3, 3), 3)
            refs = []
            for structure in structures:
                payload = [a.tolist() for a in structure.layers]
                refs.append({'payload': payload, 'key': voxel_key(payload),
                             'complexity': sum(cell for layer in payload for row in layer for cell in row)})
            assert len(refs) == 3 ** m
            # Independently enumerate nested layer subsets, including trailing zero layers.
            masks = list(range(512))
            bottom = int(top, 2)
            independent = set()
            for second in masks:
                if second & bottom != second:
                    continue
                for third in masks:
                    if third & second != third:
                        continue
                    layers = [[[int(c) for c in format(mask, '09b')[i:i+3]] for i in (0, 3, 6)]
                              for mask in (bottom, second, third)]
                    independent.add(voxel_key(layers))
            assert independent == {r['key'] for r in refs}
            cases.append({'task': '3d', 'difficulty': difficulty, 'case_id': f'3d_{difficulty}_{top}',
                          'observation': top, 'references': refs,
                          'construction': {'grid_size': 3, 'max_height': 3, 'occupied_columns': m,
                                           'view': 'single_top_projection', 'seed': 33550336, 'population': 36 if m == 2 else 84}})
    for n, counts in [(5, (3, 4, 5)), (6, (5, 6, 7))]:
        for k in counts:
            cases.append(causal_case(n, k, 33550336 + 100*n + k))
    audits = []
    for case in cases:
        adapter = adapter_for(case)
        for ref in case['references']:
            verdict = adapter.evaluate(ref['payload'])
            assert verdict['valid'] and verdict['key'] == ref['key'] and verdict['complexity'] == ref['complexity'], case['case_id']
            # Cross-check every causal truth against preserved upstream forward simulation.
            if case['task'] == 'causal':
                obj = adapter.module.CausalGraph.from_dict(ref['payload'])
                assert adapter.runner.validate_hypothesis(obj, case['observations'])
        truth = {r['key'] for r in case['references']}
        assert len(truth) == len(case['references'])
        case['M'] = len(truth)
        audits.append({'case_id': case['case_id'], 'task': case['task'], 'difficulty': case['difficulty'],
                       'M': len(truth), 'all_references_valid': True, 'reference_unique': True,
                       'reference_complete': True, 'complexity_counts': dict(sorted(Counter(r['complexity'] for r in case['references']).items()))})
    return cases, libraries, audits
