"""Offline transport and full experiment regression checks; no paid calls."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'shared'))
from experiment_provider import configuration, configure_environment, ExperimentClient, LoggedClient
from compositional.llm import Client
from experiment_cli import configure_engine, prepare, report
from io_utils import write, sha


def response(content='answer', usage=None):
    return io.StringIO(json.dumps(dict(model='returned-model', provider='returned-provider',
        choices=[dict(message=dict(content=content), finish_reason='stop')],
        usage=usage if usage is not None else dict(prompt_tokens=10, completion_tokens=5, total_tokens=15))))


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'TEST_UNIFIED_KEY': 'fake-test-secret'})
        self.env.start()
        for key in list(os.environ):
            if key.startswith('EXPERIMENT_'):
                os.environ.pop(key)

    def tearDown(self):
        self.env.stop()

    def client(self, provider, thinking='provider_default'):
        config = configuration(provider, 'explicit-model', key_env='TEST_UNIFIED_KEY', thinking=thinking)
        configure_environment(config)
        return ExperimentClient(config['model'], config['base_url'], config['key_env'])

    def test_generic_request_preserves_original_interface(self):
        captured = []
        def send(req, **kwargs):
            captured.append((req.full_url, req.headers, json.loads(req.data)))
            return response()
        with patch('urllib.request.urlopen', side_effect=send):
            Client('explicit-model', key_env='TEST_UNIFIED_KEY').query('same prompt')
            self.client('compatible').query('same prompt')
        self.assertEqual(captured[0], captured[1])

    def test_provider_payloads_and_unknown_costs(self):
        for provider in ('deepseek', 'openrouter'):
            for thinking in ('provider_default', 'enabled', 'disabled'):
                client = self.client(provider, thinking)
                body = client.request_body('prompt')
                with patch('urllib.request.urlopen', return_value=response()) as send:
                    self.assertEqual(client.query('prompt'), 'answer')
                self.assertEqual(send.call_count, 1)
                self.assertEqual(client.calls, 1)
                self.assertIsNone(client.report()['estimated_cost_usd'])
                if provider == 'deepseek':
                    self.assertNotIn('reasoning', body)
                    self.assertEqual(body.get('thinking'), None if thinking == 'provider_default' else {'type': thinking})
                else:
                    self.assertNotIn('thinking', body)
                    self.assertFalse(body['provider']['allow_fallbacks'])
                    self.assertEqual(body.get('reasoning'), None if thinking == 'provider_default' else {'enabled': thinking == 'enabled'})

    def test_one_failed_request_consumes_one_call_without_secrets(self):
        client = self.client('deepseek', 'disabled')
        error = urllib.error.HTTPError(client.base_url, 429, 'fake-test-secret', {}, None)
        with patch('urllib.request.urlopen', side_effect=error) as send:
            with self.assertRaises(RuntimeError): client.query('prompt')
        self.assertEqual((send.call_count, client.calls), (1, 1))
        self.assertEqual(client.errors[0]['status'], 429)
        self.assertNotIn('fake-test-secret', json.dumps(client.report()))

    def test_log_provenance_usage_and_failure(self):
        config = configuration('openrouter', 'explicit-model', key_env='TEST_UNIFIED_KEY', thinking='disabled')
        configure_environment(config)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'call.jsonl'
            client = LoggedClient(config['model'], config['base_url'], config['key_env']).attach(path, 'test')
            client.slot_getter = lambda: client.calls
            usage = dict(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.01,
                         completion_tokens_details=dict(reasoning_tokens=2))
            with patch('urllib.request.urlopen', return_value=response(usage=usage)):
                client.query('prompt')
            self.assertEqual(client.report()['provider_reported_cost_usd'], 0.01)
            with patch('urllib.request.urlopen', side_effect=TimeoutError()):
                with self.assertRaises(RuntimeError): client.query('again')
            self.assertIsNone(client.report()['provider_reported_cost_usd'])
            records = [json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual([r['slot'] for r in records], [1, 2])
            self.assertEqual(records[0]['usage'], usage)
            self.assertEqual(records[0]['response_provider'], 'returned-provider')
            self.assertEqual(records[1]['usage'], None)
            self.assertNotIn('fake-test-secret', path.read_text())

    def test_invalid_configuration_fails_before_request(self):
        for endpoint in ('http://localhost.evil.test', 'https://user:secret@example.org', 'https://example.org?key=secret'):
            with self.assertRaises(ValueError): configuration(model='model', base_url=endpoint)
        with self.assertRaises(ValueError): configuration()
        with self.assertRaises(ValueError): configuration(model='model', thinking='disabled')
        with self.assertRaises(ValueError): configuration(model='model', input_price=float('nan'))


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from matrix_data import generate
        with patch('urllib.request.urlopen', side_effect=AssertionError('Network prohibited')):
            cls.cases, _, _ = generate()

    def test_reference_counts_and_column_library(self):
        from column_prompt_protocol import column_blocks, make_library
        self.assertEqual(len(self.cases), 28)
        self.assertEqual(sum(c['M'] for c in self.cases), 784)
        self.assertEqual(column_blocks(3), ['000', '100', '110', '111'])
        self.assertNotIn('Mandatory support rule', make_library('3d', 'hard')['prompt'])
        self.assertEqual(sorted(c['M'] for c in self.cases if c['task'] == '3d'), [9, 9, 9, 27, 27, 27])

    def test_complete_curves_all_tasks_methods_and_budgets(self):
        from column_prompt_protocol import METHODS, make_library
        from matrix_adapters import public_case
        selected = [next(c for c in self.cases if c['task'] == t and c['difficulty'] == d)
                    for t in ('3d', 'boolean', 'causal') for d in ('medium', 'hard')]
        config = configuration('deepseek', 'offline-model', key_env='TEST_UNIFIED_KEY', thinking='disabled')
        with patch.dict(os.environ, {'TEST_UNIFIED_KEY': 'fake-test-secret'}), contextlib.redirect_stdout(io.StringIO()):
            engine = configure_engine(config)
            for multiplier in (1, 2):
                with tempfile.TemporaryDirectory() as temp:
                    folder = Path(temp)
                    libraries = {f"{c['task']}_{c['difficulty']}": make_library(c['task'], c['difficulty']) for c in selected}
                    write(folder / 'dataset.json', {'cases': selected})
                    write(folder / 'libraries.json', libraries)
                    protocol = dict(config, methods=METHODS, budget_multiplier=multiplier,
                        expected_runs=len(selected)*len(METHODS), expected_total_calls=sum(c['M']*multiplier*len(METHODS) for c in selected),
                        dataset_sha256=sha(folder / 'dataset.json'), libraries_sha256=sha(folder / 'libraries.json'), source_sha256={})
                    runs = []
                    for case in selected:
                        payload = case['references'][0]['payload']
                        if case['task'] == 'boolean': output = 'Expression: ' + payload
                        elif case['task'] == 'causal': output = 'Graph: ' + ', '.join(a+'->'+b for a,b in payload['edges'])
                        else:
                            output = 'Structure:\n' + '\n'.join('Layer '+str(i+1)+':\n'+'\n'.join(' '.join(map(str,row)) for row in layer) for i,layer in enumerate(payload))
                        for method in METHODS:
                            count = [0]
                            def send(*args, **kwargs):
                                count[0] += 1
                                if count[0] % 4 == 0: raise TimeoutError()
                                return response('invalid' if count[0] % 4 == 3 else output)
                            with patch('urllib.request.urlopen', side_effect=send):
                                result = engine.run_one(str(folder), case, method, libraries[f"{case['task']}_{case['difficulty']}"], multiplier)
                            self.assertEqual(count[0], case['M']*multiplier)
                            self.assertEqual(result['llm']['calls'], case['M']*multiplier)
                            self.assertGreater(len(result['llm']['errors']), 0)
                            runs.append(result)
                    summary = report(folder, engine, selected, protocol)
                    self.assertEqual(summary['completed_runs'], 24)
                    self.assertTrue(json.loads((folder / 'audit.json').read_text())['passed'])
                    with patch('urllib.request.urlopen', side_effect=AssertionError('Must not replay')):
                        engine.run_one(str(folder), selected[0], METHODS[0], libraries[f"{selected[0]['task']}_{selected[0]['difficulty']}"], multiplier)

    def test_prepare_provider_freeze_and_no_network(self):
        config = configuration('openrouter', 'explicit-model')
        with tempfile.TemporaryDirectory() as temp, patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')):
            folder = Path(temp)
            _, _, first = prepare(folder, config, ['3d'], ['medium', 'hard'], ['prompt_library'], 1)
            self.assertEqual(first['expected_total_calls'], 108)
            prepare(folder, config, ['3d'], ['medium', 'hard'], ['prompt_library'], 1)
            with self.assertRaises(ValueError):
                prepare(folder, configuration('deepseek', 'explicit-model'), ['3d'], ['medium', 'hard'], ['prompt_library'], 1)

    def test_task_cli_runs_from_external_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            env = {k:v for k,v in os.environ.items() if not k.startswith('EXPERIMENT_') and k not in ('DEEPSEEK_API_KEY', 'OPENROUTER_API_KEY', 'OPENAI_API_KEY')}
            result = subprocess.run([sys.executable, '-B', str(ROOT / '3d/run_experiment.py'),
                '--provider', 'deepseek', '--model', 'offline-placeholder', '--thinking', 'disabled',
                '--output', str(Path(temp)/'prepared')], cwd=temp, env=env, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            protocol = json.loads((Path(temp)/'prepared/protocol.json').read_text(encoding='utf-8'))
            self.assertEqual(protocol['expected_total_calls'], 432)
            self.assertEqual(protocol['budget_multiplier'], 1)
            self.assertEqual(protocol['provider'], 'deepseek')
            self.assertFalse((Path(temp)/'prepared/calls').exists())


if __name__ == '__main__':
    unittest.main()
