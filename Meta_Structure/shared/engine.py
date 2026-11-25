"""Preserved column experiment loops and metrics; task-independent shared engine."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from experiment_provider import LoggedClient, MODEL, BASE_URL, KEY_ENV
from io_utils import write, sha
from compositional.upstream import ROOT
from matrix_adapters import adapter_for, compact, public_case, \
    upstream_error_text, upstream_error_detail, upstream_history_add
from column_prompt_protocol import METHODS, STRATIFIED_METHODS, LIBRARY_METHODS, build_prompt

MAX_RETRIES = {'boolean': 5, '3d': 3, 'causal': 3}


def csv_write(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: compact(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def schedule_for(case):
    counts = Counter(r['complexity'] for r in case['references'])
    cycle = [c for c in sorted(counts) for _ in range(counts[c])]
    assert len(cycle) == case['M']
    return cycle + cycle


def history_add(task, history, seen, payload):
    if payload is None:
        return
    if task == '3d':
        key = compact(payload)
        if key in seen:
            return
        seen.add(key)
    history.append(payload)


class CurveAccumulator:
    def __init__(self, case):
        self.case = case
        self.truth = {r['key']: r['complexity'] for r in case['references']}
        self.valid_count = 0
        self.seen_raw, self.seen_space, self.seen_valid, self.recovered = set(), set(), set(), set()
        self.failures = Counter()
        self.count = 0

    def add(self, payload, verdict, api_error=False):
        self.count += 1
        if api_error:
            self.failures['api_errors'] += 1
        if payload is None:
            self.failures['parse_failures'] += 1
        exact_duplicate = False
        canonical_duplicate = False
        if payload is not None:
            raw_key = compact(payload)
            exact_duplicate = raw_key in self.seen_raw
            self.seen_raw.add(raw_key)
        if not verdict['in_space']:
            self.failures['out_of_space'] += 1
        if verdict['in_space']:
            canonical_duplicate = verdict['key'] in self.seen_space
            self.seen_space.add(verdict['key'])
        if exact_duplicate:
            self.failures['exact_duplicates'] += 1
        if canonical_duplicate:
            self.failures['canonical_duplicates'] += 1
        if verdict['in_space'] and not verdict['valid']:
            self.failures['observation_mismatches'] += 1
        new_valid = verdict['valid'] and verdict['key'] not in self.seen_valid
        if verdict['valid']:
            self.valid_count += 1
            self.seen_valid.add(verdict['key'])
            if verdict['key'] in self.truth:
                self.recovered.add(verdict['key'])
            else:
                self.failures['valid_outside_reference'] += 1
        hist = Counter(self.truth[k] for k in self.recovered)
        return {'calls': self.count, 'calls_over_M': self.count / self.case['M'],
                'n_proposals': self.count, 'n_valid': self.valid_count, 'n_unique_in_space': len(self.seen_space),
                'n_unique_valid': len(self.seen_valid), 'n_recovered': len(self.recovered),
                'VR': self.valid_count/self.count, 'NR': len(self.seen_space)/self.count,
                'RR': len(self.recovered)/self.case['M'], 'new_valid': bool(new_valid),
                'exact_duplicate': exact_duplicate, 'canonical_duplicate': canonical_duplicate,
                'api_errors': self.failures['api_errors'], 'parse_failures': self.failures['parse_failures'],
                'out_of_space': self.failures['out_of_space'], 'observation_mismatches': self.failures['observation_mismatches'],
                'exact_duplicate_count': self.failures['exact_duplicates'], 'canonical_duplicate_count': self.failures['canonical_duplicates'],
                'valid_outside_reference': self.failures['valid_outside_reference'], 'recovered_by_complexity': dict(sorted(hist.items()))}


def run_one_upstream(folder_text, case, budget_multiplier=2):
    """Replicate the upstream baseline loop: round budget, per-task retries,
    verbatim create_prompt output, and upstream prior-history semantics."""
    if budget_multiplier not in (1, 2):
        raise ValueError('Budget multiplier must be 1 or 2')
    budget = budget_multiplier * case['M']
    folder = Path(folder_text)
    stem = case['case_id'] + '__history_baseline'
    destination = folder / 'runs' / (stem + '.json')
    log = folder / 'calls' / (stem + '.jsonl')
    curve_path = folder / 'curves' / (stem + '.jsonl')
    journal = folder / 'journals' / (stem + '.jsonl')
    if destination.exists():
        saved = json.loads(destination.read_text(encoding='utf-8'))
        if saved['budget'] != budget:
            raise ValueError('Saved result has a different budget')
        return saved
    if any(p.exists() for p in [log, curve_path, journal]):
        raise RuntimeError('Partial run found; inspect instead of replaying paid calls: ' + stem)
    for path in [destination, log, curve_path, journal]:
        path.parent.mkdir(parents=True, exist_ok=True)
    adapter = adapter_for(public_case(case))
    client = LoggedClient(MODEL, BASE_URL, KEY_ENV, 0.7, 2048).attach(log, stem)
    accumulator = CurveAccumulator(case)
    history, history_seen, proposals, curves = [], set(), [], []
    max_retries = MAX_RETRIES[case['task']]
    start = time.perf_counter()
    print(f"START {stem} M={case['M']} rounds={budget} max_retries={max_retries}", flush=True)
    for index in range(budget):
        client.slot_getter = lambda: index + 1
        prompt = adapter.upstream_prompt(history)
        with journal.open('a', encoding='utf-8') as handle:
            handle.write(compact({'call': index+1, 'event': 'request_start', 'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        response, payload, api_error, error_detail = None, None, False, None
        attempts = 0
        for attempt in range(max_retries):
            attempts += 1
            try:
                response = client.query(prompt)
            except RuntimeError:
                api_error = True
                error_detail = upstream_error_detail(adapter, max_retries,
                                                     upstream_error_text(client.errors[-1]))
                continue
            payload = adapter.parse(response)
            if payload is not None:
                break
        verdict = adapter.evaluate(payload)
        upstream_history_add(adapter, case, history, history_seen, payload, error_detail)
        proposal = {'index': index, 'call': index+1, 'payload': payload, 'attempts': attempts,
                    'api_error': api_error, **verdict}
        proposals.append(proposal)
        curve = accumulator.add(payload, verdict, api_error)
        report = client.report()
        curve.update({'task': case['task'], 'difficulty': case['difficulty'], 'case_id': case['case_id'],
            'method': 'history_baseline', 'M': case['M'], 'cumulative_prompt_tokens': client.usage['prompt_tokens'],
            'cumulative_completion_tokens': client.usage['completion_tokens'], 'cumulative_total_tokens': client.usage['total_tokens'],
            'cumulative_estimated_cost_usd': report['estimated_cost_usd'], 'elapsed_seconds': time.perf_counter()-start,
            'last_finish_reason': client.records[-1].get('finish_reason', 'error'),
            'last_usage': client.last_usage, 'target_complexity': None,
            'target_match_count': None, 'payload_valid': verdict['valid'],
            'payload_in_space': verdict['in_space'], 'payload_complexity': verdict['complexity'],
            'attempts': attempts, 'max_retries': max_retries})
        curves.append(curve)
        with curve_path.open('a', encoding='utf-8') as handle:
            handle.write(compact(curve) + '\n')
        if curve['valid_outside_reference']:
            write(folder / 'audit_failure.json', {'case_id': case['case_id'], 'method': 'history_baseline', 'proposal': proposal})
            raise RuntimeError('Valid result outside frozen reference set')
    assert len(proposals) == budget
    checkpoints = {'1M': curves[case['M']-1]}
    if budget_multiplier == 2:
        checkpoints['2M'] = curves[-1]
    result = {'task': case['task'], 'difficulty': case['difficulty'], 'case_id': case['case_id'], 'method': 'history_baseline',
        'M': case['M'], 'complete': True, 'budget': budget, 'llm': client.report(), 'proposals': proposals,
        'checkpoints': checkpoints, 'elapsed_seconds': time.perf_counter()-start,
        'max_retries': max_retries, 'replication': 'upstream_rounds'}
    write(destination, result)
    csv_write(folder / 'curves' / (stem + '.csv'), curves)
    print(f"DONE {stem} " + ' '.join(f"RR@{tier}={value['RR']:.4f}" for tier, value in checkpoints.items()), flush=True)
    return result


def run_one(folder_text, case, method, library, budget_multiplier=2):
    if method == 'history_baseline':
        return run_one_upstream(folder_text, case, budget_multiplier)
    if method not in METHODS:
        raise ValueError('Unknown method: ' + method)
    if budget_multiplier not in (1, 2):
        raise ValueError('Budget multiplier must be 1 or 2')
    budget = budget_multiplier * case['M']
    folder = Path(folder_text)
    stem = case['case_id'] + '__' + method
    destination = folder / 'runs' / (stem + '.json')
    log = folder / 'calls' / (stem + '.jsonl')
    curve_path = folder / 'curves' / (stem + '.jsonl')
    journal = folder / 'journals' / (stem + '.jsonl')
    if destination.exists():
        saved = json.loads(destination.read_text(encoding='utf-8'))
        if saved['budget'] != budget:
            raise ValueError('Saved result has a different budget')
        return saved
    if any(p.exists() for p in [log, curve_path, journal]):
        raise RuntimeError('Partial run found; inspect instead of replaying paid calls: ' + stem)
    for path in [destination, log, curve_path, journal]:
        path.parent.mkdir(parents=True, exist_ok=True)
    adapter = adapter_for(public_case(case))
    client = LoggedClient(MODEL, BASE_URL, KEY_ENV, 0.7, 2048).attach(log, stem)
    client.slot_getter = lambda: client.calls
    schedule = schedule_for(case) if method in STRATIFIED_METHODS else None
    accumulator = CurveAccumulator(case)
    history, history_seen, proposals, curves = [], set(), [], []
    target_hits = 0
    start = time.perf_counter()
    print(f"START {stem} M={case['M']} calls={budget}", flush=True)
    for index in range(budget):
        prompt = build_prompt(adapter, history, method, library, schedule[index] if schedule is not None else None)
        with journal.open('a', encoding='utf-8') as handle:
            handle.write(compact({'call': index+1, 'event': 'request_start', 'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        response, api_error = None, False
        try:
            response = client.query(prompt)
        except RuntimeError:
            api_error = True
        payload = adapter.parse(response) if response is not None else None
        verdict = adapter.evaluate(payload)
        history_add(case['task'], history, history_seen, payload)
        proposal = {'index': index, 'call': index+1, 'payload': payload, **verdict}
        if schedule is not None:
            matched = verdict['in_space'] and verdict['complexity'] == schedule[index]
            target_hits += int(matched)
            proposal.update({'target_complexity': schedule[index], 'target_matched': bool(matched)})
        proposals.append(proposal)
        curve = accumulator.add(payload, verdict, api_error)
        report = client.report()
        curve.update({'task': case['task'], 'difficulty': case['difficulty'], 'case_id': case['case_id'],
            'method': method, 'M': case['M'], 'cumulative_prompt_tokens': client.usage['prompt_tokens'],
            'cumulative_completion_tokens': client.usage['completion_tokens'], 'cumulative_total_tokens': client.usage['total_tokens'],
            'cumulative_estimated_cost_usd': report['estimated_cost_usd'], 'elapsed_seconds': time.perf_counter()-start,
            'last_finish_reason': client.records[-1].get('finish_reason', 'error'),
            'last_usage': client.last_usage, 'target_complexity': schedule[index] if schedule else None,
            'target_match_count': target_hits if schedule else None, 'payload_valid': verdict['valid'],
            'payload_in_space': verdict['in_space'], 'payload_complexity': verdict['complexity']})
        curves.append(curve)
        with curve_path.open('a', encoding='utf-8') as handle:
            handle.write(compact(curve) + '\n')
        if curve['valid_outside_reference']:
            write(folder / 'audit_failure.json', {'case_id': case['case_id'], 'method': method, 'proposal': proposal})
            raise RuntimeError('Valid result outside frozen reference set')
    assert len(proposals) == client.calls == budget
    checkpoints = {'1M': curves[case['M']-1]}
    if budget_multiplier == 2:
        checkpoints['2M'] = curves[-1]
    result = {'task': case['task'], 'difficulty': case['difficulty'], 'case_id': case['case_id'], 'method': method,
        'M': case['M'], 'complete': True, 'budget': budget, 'llm': client.report(), 'proposals': proposals,
        'checkpoints': checkpoints, 'elapsed_seconds': time.perf_counter()-start}
    write(destination, result)
    csv_write(folder / 'curves' / (stem + '.csv'), curves)
    print(f"DONE {stem} " + ' '.join(f"RR@{tier}={value['RR']:.4f}" for tier, value in checkpoints.items()), flush=True)
    return result


def summarize(folder, runs, protocol):
    rows, endpoints = [], []
    for run in runs:
        for budget, value in run['checkpoints'].items():
            endpoints.append({**value, 'budget_tier': budget})
    for task in ['boolean', '3d', 'causal']:
        for level in ['medium', 'hard']:
            for tier in ['1M', '2M']:
                for method in METHODS:
                    group = [row for row in endpoints if row['task'] == task and row['difficulty'] == level
                             and row['budget_tier'] == tier and row['method'] == method]
                    if not group:
                        continue
                    costs = [r['cumulative_estimated_cost_usd'] for r in group]
                    row = {'task': task, 'difficulty': level, 'budget_tier': tier, 'method': method,
                           'n_cases': len(group), 'calls': sum(r['calls'] for r in group),
                           'prompt_tokens': sum(r['cumulative_prompt_tokens'] for r in group),
                           'completion_tokens': sum(r['cumulative_completion_tokens'] for r in group),
                           'estimated_cost_usd': sum(costs) if all(v is not None for v in costs) else None,
                           'n_recovered': sum(r['n_recovered'] for r in group), 'sum_M': sum(r['M'] for r in group)}
                    for metric in ['VR', 'NR', 'RR']:
                        row[metric] = statistics.mean(r[metric] for r in group)
                        row[metric + '_std'] = statistics.pstdev(r[metric] for r in group)
                    rows.append(row)
    result = {'completed_runs': len(runs), 'expected_runs': protocol['expected_runs'], 'summary': rows,
              'protocol_file': 'protocol.json', 'planned_total_calls': protocol['expected_total_calls']}
    write(folder / 'summary.json', result)
    csv_write(folder / 'summary.csv', rows)
    csv_write(folder / 'endpoints.csv', endpoints)
    return result


def final_audit(folder, cases, runs, protocol):
    by_id = {c['case_id']: c for c in cases}
    assert len(runs) == protocol['expected_runs']
    assert {(r['case_id'], r['method']) for r in runs} == {(c['case_id'], m) for c in cases for m in protocol['methods']}
    assert sha(folder/'dataset.json') == protocol['dataset_sha256']
    assert sha(folder/'libraries.json') == protocol['libraries_sha256']
    checks, all_curves = [], []
    for relative, expected in protocol['source_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    libraries = json.loads((folder / 'libraries.json').read_text(encoding='utf-8'))
    for run in runs:
        stem = run['case_id'] + '__' + run['method']
        records = [json.loads(line) for line in (folder/'calls'/(stem+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        curves = [json.loads(line) for line in (folder/'curves'/(stem+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        case = by_id[run['case_id']]
        assert run['budget'] == protocol['budget_multiplier'] * case['M']
        accumulator = CurveAccumulator(case)
        adapter = adapter_for(public_case(case))
        replay_history, replay_seen = [], set()
        fixed = libraries[f"{case['task']}_{case['difficulty']}"]['prompt']
        expected_schedule = schedule_for(case)
        cumulative = Counter()
        if run['method'] == 'history_baseline':
            # Replicate the upstream round loop from saved records: one prompt per round,
            # repeated across retries, with upstream prior-history semantics.
            assert len(curves) == run['budget'] == len(run['proposals'])
            assert len(records) == run['llm']['calls'] >= run['budget']
            assert run['max_retries'] == MAX_RETRIES[case['task']]
            by_slot = {}
            for record in records:
                by_slot.setdefault(record['slot'], []).append(record)
            assert sorted(by_slot) == list(range(1, run['budget'] + 1))
            for index, (proposal, curve) in enumerate(zip(run['proposals'], curves)):
                round_records = by_slot[index + 1]
                assert 1 <= len(round_records) <= run['max_retries']
                assert len(round_records) == proposal['attempts'] == curve['attempts']
                expected_prompt = adapter.upstream_prompt(replay_history)
                assert all(record['prompt'] == expected_prompt for record in round_records)
                payload = None
                for record in round_records:
                    if 'response' in record:
                        payload = adapter.parse(record['response'])
                        if payload is not None:
                            break
                assert json.loads(compact(payload)) == json.loads(compact(proposal['payload']))
                verdict = adapter.evaluate(payload)
                for k, v in verdict.items():
                    assert proposal[k] == v
                api_error = any('error' in record for record in round_records)
                assert proposal['api_error'] == api_error
                error_detail = None
                if api_error:
                    last_error = [r for r in round_records if 'error' in r][-1]['error']
                    error_detail = upstream_error_detail(adapter, run['max_retries'], upstream_error_text(last_error))
                upstream_history_add(adapter, case, replay_history, replay_seen, payload, error_detail)
                independently = accumulator.add(payload, verdict, api_error)
                for metric, value in independently.items():
                    assert json.loads(compact(curve[metric])) == json.loads(compact(value))
                for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                    for record in round_records:
                        cumulative[key] += (record.get('usage') or {}).get(key, 0)
                    assert curve['cumulative_' + key] == cumulative[key]
        else:
            assert len(records) == len(curves) == run['llm']['calls'] == run['budget'] == len(run['proposals'])
            for index, (record, proposal, curve) in enumerate(zip(records, run['proposals'], curves)):
                assert record['call'] == proposal['call'] == curve['calls'] == index+1
                assert record['slot'] == index+1
                target = expected_schedule[index] if run['method'] in STRATIFIED_METHODS else None
                expected_prompt = build_prompt(adapter, replay_history, run['method'],
                    libraries[f"{case['task']}_{case['difficulty']}"], target)
                assert record['prompt'] == expected_prompt
                assert record['prompt'].endswith(fixed) == (run['method'] in LIBRARY_METHODS)
                if run['method'] in STRATIFIED_METHODS:
                    assert curve['target_complexity'] == expected_schedule[index]
                payload = adapter.parse(record['response']) if 'response' in record else None
                assert json.loads(compact(payload)) == json.loads(compact(proposal['payload']))
                verdict = adapter.evaluate(payload)
                for k, v in verdict.items():
                    assert proposal[k] == v
                independently = accumulator.add(payload, verdict, 'error' in record)
                for metric, value in independently.items():
                    assert json.loads(compact(curve[metric])) == json.loads(compact(value))
                history_add(case['task'], replay_history, replay_seen, payload)
                for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                    cumulative[key] += (record.get('usage') or {}).get(key, 0)
                    assert curve['cumulative_' + key] == cumulative[key]
        for key in cumulative:
            assert cumulative[key] == run['llm']['usage'][key]
        assert json.loads(compact(run['checkpoints']['1M'])) == curves[case['M']-1]
        if protocol['budget_multiplier'] == 2:
            assert json.loads(compact(run['checkpoints']['2M'])) == curves[-1]
        assert not any(r['valid_outside_reference'] for r in curves)
        all_curves.extend(curves)
        checks.append({'case_id': run['case_id'], 'method': run['method'], 'calls': len(records),
                       'finish_reasons': run['llm']['finish_reason_counts'], 'api_errors': len(run['llm']['errors']),
                       'missing_usage': run['llm']['usage_missing_calls']})
    assert sum(r['budget'] for r in runs) == protocol['expected_total_calls']
    assert sum(r['llm']['calls'] for r in runs) >= protocol['expected_total_calls']
    write(folder/'audit.json', {'passed': True, 'source_hashes_unchanged': True, 'curves_recomputed_from_raw_responses': True,
          'proposals_equal_rounds': True, 'retries_within_limits': True, 'fixed_libraries_verified': True, 'token_sums_match': True, 'checks': checks})
    csv_write(folder/'all_call_curves.csv', sorted(all_curves, key=lambda r: (r['task'], r['difficulty'], r['case_id'], r['method'], r['calls'])))
    return all_curves
