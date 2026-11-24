"""Task-oriented entrypoint for frozen, equally budgeted comparisons."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random

import yaml

from experiment_provider import configuration, configure_environment
from io_utils import freeze, sha, write

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['history_baseline', 'paper_stratified', 'prompt_library', 'stratified_library']
TASKS = ['3d', 'boolean', 'causal']


def source_hashes():
    files = list(ROOT.glob('*.py')) + list(ROOT.glob('requirements*.txt'))
    for directory in ['shared', *TASKS]:
        files += [p for p in (ROOT / directory).rglob('*') if p.suffix in ('.py', '.yaml')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def prepare(folder, config, tasks, levels, methods, multiplier):
    from matrix_data import generate
    from column_prompt_protocol import make_library
    cases, _, checks = generate()
    cases = [c for c in cases if c['task'] in tasks and c['difficulty'] in levels]
    cases.sort(key=lambda c: c['case_id'])
    for case in cases:
        case['references'].sort(key=lambda r: r['key'])
    libraries = {f'{task}_{level}': make_library(task, level) for task in tasks for level in levels}
    jobs = [(c['case_id'], m) for c in cases for m in methods]
    random.Random(33550336).shuffle(jobs)
    dataset = {'cases': cases}
    # Compare everything before freezing any file, so a mismatched reuse cannot partially update inputs.
    signature = dict(version='unified_columns_v1', **config, tasks=tasks, difficulties=levels,
                     methods=methods, budget_multiplier=multiplier, temperature=0.7, max_tokens=2048,
                     source_sha256=source_hashes(), expected_runs=len(jobs), job_order=jobs,
                     expected_total_calls=sum(c['M'] * multiplier * len(methods) for c in cases),
                     paper_exact_replication=False, prompt_version='vertical_columns_v2',
                     generation='one whole model answer per API request; no retries or completion by code',
                     budget='M is admissible reference count; 2M includes nested M checkpoint',
                     strata='oracle complexity counts, no reference answers in prompts',
                     boolean_operator_order='sorted for reproducible prompts across processes',
                     cost_basis='optional configured USD/million token estimate; raw provider usage retained')
    candidate = json.loads(json.dumps(signature))
    if (folder / 'protocol.json').exists():
        old = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
        old.pop('dataset_sha256', None); old.pop('libraries_sha256', None)
        if old != candidate:
            raise ValueError('Provider, parameters or source changed; use a new output directory')
    elif any(folder.iterdir()):
        raise ValueError('Output directory must be empty for a new experiment')
    freeze(folder / 'dataset.json', dataset)
    freeze(folder / 'libraries.json', libraries)
    signature.update(dataset_sha256=sha(folder / 'dataset.json'), libraries_sha256=sha(folder / 'libraries.json'))
    freeze(folder / 'protocol.json', signature)
    ids = {c['case_id'] for c in cases}
    freeze(folder / 'preflight.json', dict(passed=True, api_calls=0, checks=[c for c in checks if c['case_id'] in ids]))
    return cases, libraries, signature


def configure_engine(config):
    configure_environment(config)
    import engine
    # Also supports several offline checks in one interpreter.
    engine.MODEL, engine.BASE_URL, engine.KEY_ENV = config['model'], config['base_url'], config['key_env']
    return engine


def report(folder, engine, cases, protocol):
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder / 'runs').glob('*.json'))]
    if len(runs) != protocol['expected_runs']:
        raise ValueError('Experiment incomplete; existing summary remains partial')
    # Validate transport provenance as well as recomputing original model-output metrics.
    for path in (folder / 'calls').glob('*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            for field in ('provider', 'base_url', 'thinking'):
                if item.get(field) != protocol[field]:
                    raise ValueError('Call provenance differs from frozen protocol')
            if item['request']['model'] != protocol['model']:
                raise ValueError('Requested model differs from frozen protocol')
    engine.final_audit(folder, cases, runs, protocol)
    return engine.summarize(folder, runs, protocol)


def main(task=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='YAML defaults; command-line options override them')
    parser.add_argument('--provider', choices=['compatible', 'deepseek', 'openrouter'])
    parser.add_argument('--model', help='Exact model ID accepted by the chosen provider')
    parser.add_argument('--base-url')
    parser.add_argument('--key-env')
    parser.add_argument('--thinking', choices=['provider_default', 'enabled', 'disabled'])
    parser.add_argument('--input-price', type=float, help='USD per million input tokens; omitted means unknown')
    parser.add_argument('--output-price', type=float, help='USD per million output tokens; omitted means unknown')
    parser.add_argument('--tasks', choices=TASKS, nargs='+')
    parser.add_argument('--difficulty', choices=['medium', 'hard'], nargs='+')
    parser.add_argument('--methods', choices=METHODS, nargs='+')
    parser.add_argument('--budget-multiplier', type=int, choices=[1, 2])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Enable paid API calls; default only prepares local inputs')
    mode.add_argument('--report-only', action='store_true', help='Audit complete saved results without API calls')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    folder = args.output.resolve()
    if args.report_only:
        protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
        if protocol.get('version') != 'unified_columns_v1':
            raise ValueError('Use the original reporter for historical experiment formats')
        config = {key: protocol[key] for key in ('provider', 'model', 'base_url', 'key_env', 'thinking', 'input_price', 'output_price')}
        engine = configure_engine(config)
        cases = json.loads((folder / 'dataset.json').read_text(encoding='utf-8'))['cases']
        report(folder, engine, cases, protocol)
        print('AUDIT PASS: ' + str(folder))
        return
    path = args.config or (ROOT / task / 'config/experiment.yaml' if task else None)
    defaults = yaml.safe_load(path.read_text(encoding='utf-8')) if path else {}
    if not isinstance(defaults, dict):
        parser.error('Configuration must be a YAML mapping')
    def value(name, fallback=None):
        supplied = getattr(args, name)
        return supplied if supplied is not None else defaults.get(name, fallback)
    config = configuration(**{k: value(k, {'provider': 'compatible', 'thinking': 'provider_default'}.get(k))
                              for k in ('provider', 'model', 'base_url', 'key_env', 'thinking', 'input_price', 'output_price')})
    tasks = value('tasks', [task] if task else TASKS)
    levels = value('difficulty', ['medium', 'hard'])
    methods = value('methods', METHODS)
    multiplier = value('budget_multiplier', 1)
    for label, selected, allowed in [('tasks', tasks, TASKS), ('difficulty', levels, ['medium', 'hard']), ('methods', methods, METHODS)]:
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
            parser.error('Invalid or duplicated ' + label)
    if task and tasks != [task]:
        parser.error('This entry is restricted to task ' + task)
    if type(multiplier) is not int or multiplier not in (1, 2):
        parser.error('budget_multiplier must be 1 or 2')
    engine = configure_engine(config)
    if args.run:
        # Validate credentials before preparing or scheduling paid work.
        from experiment_provider import ExperimentClient
        ExperimentClient(config['model'], config['base_url'], config['key_env'])
    folder.mkdir(parents=True, exist_ok=True)
    cases, libraries, protocol = prepare(folder, config, tasks, levels, methods, multiplier)
    print(f"PREPARED cases={len(cases)} runs={protocol['expected_runs']} planned_calls={protocol['expected_total_calls']} api_calls=0")
    if not args.run:
        return
    by_id = {c['case_id']: c for c in cases}
    # Fail before starting any requests if an interrupted attempt would otherwise be replayed.
    runs = []
    pending = []
    for case_id, method in protocol['job_order']:
        stem = case_id + '__' + method
        saved = folder / 'runs' / (stem + '.json')
        if saved.exists():
            result = json.loads(saved.read_text(encoding='utf-8'))
            if result['budget'] != by_id[case_id]['M'] * multiplier or not result['complete']:
                raise ValueError('Saved budget or completion state differs')
            runs.append(result)
        else:
            if any((folder / d / (stem + '.jsonl')).exists() for d in ('calls', 'curves', 'journals')):
                raise ValueError('Interrupted run found; inspect it before creating another experiment')
            pending.append((case_id, method))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(engine.run_one, str(folder), by_id[c], m,
                              libraries[f"{by_id[c]['task']}_{by_id[c]['difficulty']}"], multiplier)
                   for c, m in pending]
        for future in as_completed(futures):
            runs.append(future.result())
            engine.summarize(folder, runs, protocol)
    report(folder, engine, cases, protocol)
    print('COMPLETE: ' + str(folder))
