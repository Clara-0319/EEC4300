"""Isolated shared semantics for the approved Boolean control experiment.

The public language is the original full, primitive-depth-2 enumerated language.
Membership is compiled from the observation-independent grammar, never the case's
answer set. Original repository files and the composition scheduler are untouched.
"""
import re
from functools import lru_cache

import sympy as sp

from settings import MECHANISTIC
from compositional.upstream import BooleanAdapter, load

METADATA = {'variables': ['x', 'y'], 'operators': ['AND', 'OR', 'NOT', 'NOR'],
            'max_depth': 2, 'mechanistic_opts': MECHANISTIC}
RULES = ('Shared language: x,y; binary AND/OR/NOR and unary NOT; primitive-gate depth at most 2. '
         'NOR(a,b), including its NOT(a OR b) spelling, is ONE primitive gate for depth. '
         'Equivalent spellings of a member of this bounded canonical language are accepted. '
         'For complexity ONLY, expand NOR into NOT plus OR and count operator nodes after '
         'the repository structural canonicalization (each flattened AND/OR counts once). '
         'Do not use constants. No reference expressions are provided.')


def parse_tree(text):
    if not isinstance(text, str) or len(text) > 12000:
        raise ValueError('invalid input')
    tokens = re.findall(r'AND|NOR|NOT|OR|x|y|[(),]|\S', text)
    if not tokens or len(tokens) > 1024:
        raise ValueError('invalid size')
    index = 0

    def accept(token):
        nonlocal index
        if index < len(tokens) and tokens[index] == token:
            index += 1
            return True
        return False

    def require(token):
        if not accept(token):
            raise ValueError('expected ' + token)

    def atom():
        nonlocal index
        if accept('NOT'):
            return ('NOT', atom())
        if index < len(tokens) and tokens[index] in ('AND', 'OR', 'NOR'):
            name = tokens[index]
            index += 1
            require('(')
            left = expression()
            require(',')
            right = expression()
            require(')')
            return (name, left, right)
        if accept('('):
            value = expression()
            require(')')
            return value
        if index < len(tokens) and tokens[index] in ('x', 'y'):
            name = tokens[index]
            index += 1
            return ('VAR', name)
        raise ValueError('unexpected token')

    def conjunction():
        value = atom()
        while accept('AND'):
            value = ('AND', value, atom())
        return value

    def expression():
        value = conjunction()
        while accept('OR'):
            value = ('OR', value, conjunction())
        return value

    result = expression()
    if index != len(tokens):
        raise ValueError('trailing tokens')
    return result


def evaluate_tree(tree, inputs):
    op = tree[0]
    if op == 'VAR':
        return inputs[tree[1]]
    if op == 'NOT':
        return 1 - evaluate_tree(tree[1], inputs)
    a, b = (evaluate_tree(child, inputs) for child in tree[1:])
    return int(a and b) if op == 'AND' else int(a or b) if op == 'OR' else int(not (a or b))


def symbolic(tree):
    op = tree[0]
    if op == 'VAR':
        return sp.Symbol(tree[1])
    children = [symbolic(child) for child in tree[1:]]
    with sp.evaluate(False):
        if op == 'NOR':
            return sp.Not(sp.Or(*children, evaluate=False), evaluate=False)
        return {'AND': sp.And, 'OR': sp.Or, 'NOT': sp.Not}[op](*children, evaluate=False)


def complexity(key):
    if key[0] in ('VAR', 'CONST'):
        return 0
    if key[0] == 'NOT':
        return 1 + complexity(key[1])
    return (2 if key[0] == 'NOR' else 1) + sum(complexity(k) for k in key[1])


@lru_cache(None)
def decode(text):
    tree = parse_tree(text)
    expr = symbolic(tree)
    if any(node in (sp.true, sp.false) for node in sp.preorder_traversal(expr)):
        raise ValueError('constant')
    key = load('boolean/boolean_dataset.py').mechanistic_key_sympy(expr, **MECHANISTIC)
    truth = {(x, y): evaluate_tree(tree, {'x': x, 'y': y}) for x in (0, 1) for y in (0, 1)}
    return key, truth


@lru_cache(None)
def language():
    data = load('boolean/boolean_dataset.py')
    pool = data.BooleanDiscoveryGame.generate_all_expressions(['x', 'y'], set(METADATA['operators']), 2, MECHANISTIC)
    result = {}
    for e in pool:
        key, truth = decode(e.formula)
        assert key == e.mechanistic_key(**MECHANISTIC)
        assert truth == e.truth_table
        result[repr(key)] = {'formula': e.formula, 'mechanistic_key': repr(key),
                            'complexity': complexity(key), 'truth_table': {str(k): v for k, v in truth.items()}}
    assert len(result) == len(pool) == 65
    return result


class ControlAdapter(BooleanAdapter):
    def __init__(self, observations):
        super().__init__(METADATA, observations)
        self.public_observations = {'observations': observations, 'shared_rules': RULES}

    def evaluate(self, payload):
        try:
            key, truth = decode(payload)
            key_text = repr(key)
            if key_text not in language():
                raise ValueError('outside bounded canonical language')
            valid = all(truth[(o.inputs['x'], o.inputs['y'])] == o.output for o in self.observations)
            return {'in_space': True, 'valid': valid, 'key': key_text,
                    'complexity': complexity(key), 'reason': 'valid' if valid else 'observation_mismatch'}
        except (ValueError, TypeError, RecursionError, KeyError):
            return {'in_space': False, 'valid': False, 'key': None, 'complexity': None, 'reason': 'parse_or_space_error'}

    def install_runner(self, schedule=None):
        # Instance-local overrides: do not monkeypatch the preserved module/class.
        original_prompt = self.runner.create_prompt
        self.prompt_index = 0

        def prompt(observations, history):
            text = original_prompt(observations, history) + '\n' + RULES
            if schedule is not None:
                target = schedule[self.prompt_index]
                text += f'\nThis proposal must have EXACTLY {target} operators under the complexity rule above. Avoid previous expressions.'
            self.prompt_index += 1
            return text

        def validate(payload, observations):
            verdict = self.evaluate(payload)
            if not verdict['in_space']:
                return False, None
            return verdict['valid'], decode(payload)[1]

        self.runner.create_prompt = prompt
        self.runner._in_space = lambda payload: self.evaluate(payload)['in_space']
        self.runner.validate_expression = validate
        self.runner.get_expression_mechanistic_key = lambda payload: decode(payload)[0] if self.evaluate(payload)['in_space'] else None
        return self.runner
