"""One Chat Completions transport; one request per proposal, never retries."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from compositional.llm import Client

PROFILES = {
    'compatible': ('https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'deepseek': ('https://api.deepseek.com', 'DEEPSEEK_API_KEY'),
    'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY'),
}
MODEL = os.environ.get('EXPERIMENT_MODEL')
PROVIDER = os.environ.get('EXPERIMENT_PROVIDER', 'compatible')
BASE_URL = os.environ.get('EXPERIMENT_BASE_URL', PROFILES.get(PROVIDER, PROFILES['compatible'])[0])
KEY_ENV = os.environ.get('EXPERIMENT_KEY_ENV', PROFILES.get(PROVIDER, PROFILES['compatible'])[1])


def configuration(provider='compatible', model=None, base_url=None, key_env=None,
                  thinking='provider_default', input_price=None, output_price=None):
    if provider not in PROFILES:
        raise ValueError('Unknown provider')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('An explicit model ID is required; no automatic model substitution')
    if thinking not in ('provider_default', 'enabled', 'disabled'):
        raise ValueError('Unknown thinking mode')
    if provider == 'compatible' and thinking != 'provider_default':
        raise ValueError('Generic endpoints require provider_default thinking')
    endpoint = (base_url or PROFILES[provider][0]).rstrip('/')
    parsed = urlsplit(endpoint)
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Endpoint must not contain credentials, query parameters or fragments')
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')):
        raise ValueError('Use HTTPS or an exact loopback host')
    for price in (input_price, output_price):
        if price is not None and (not math.isfinite(price) or price < 0):
            raise ValueError('Prices must be finite nonnegative USD per million tokens')
    return dict(provider=provider, model=model, base_url=endpoint,
                key_env=key_env or PROFILES[provider][1], thinking=thinking,
                input_price=input_price, output_price=output_price)


def configure_environment(config):
    # Child processes inherit configuration, never a secret in CLI arguments.
    for key, value in config.items():
        name = 'EXPERIMENT_' + key.upper()
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)


class ExperimentClient(Client):
    def __post_init__(self):
        for attr in ('input_price', 'output_price'):
            env = os.environ.get('EXPERIMENT_' + attr.upper())
            if getattr(self, attr) is None and env is not None:
                setattr(self, attr, float(env))
        self.config = configuration(
            os.environ.get('EXPERIMENT_PROVIDER', 'compatible'), self.model, self.base_url,
            self.key_env, os.environ.get('EXPERIMENT_THINKING', 'provider_default'),
            self.input_price, self.output_price)
        super().__post_init__()
        self.system_prompt = None
        self.last_usage = self.last_response_model = self.last_response_provider = None
        self.last_choice = self.last_response_id = None
        self.reported_cost_usd = 0.0
        self.cost_reported_calls = 0

    def request_body(self, prompt):
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        body = dict(model=self.model, messages=messages, temperature=self.temperature,
                    max_tokens=self.max_tokens)
        provider, thinking = self.config['provider'], self.config['thinking']
        if provider == 'deepseek' and thinking != 'provider_default':
            body['thinking'] = {'type': thinking}
        elif provider == 'openrouter':
            # Prevent router fallback from changing the serving provider after a failure.
            body['provider'] = {'allow_fallbacks': False, 'require_parameters': True}
            if thinking != 'provider_default':
                body['reasoning'] = {'enabled': thinking == 'enabled'}
        return body

    def query(self, prompt):
        self.calls += 1
        self.last_usage = self.last_response_model = self.last_response_provider = None
        self.last_choice = self.last_response_id = None
        request = urllib.request.Request(self.base_url.rstrip('/') + '/chat/completions',
            data=json.dumps(self.request_body(prompt)).encode(),
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + os.environ[self.key_env]}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            self.last_usage = result.get('usage')
            self.last_response_model = result.get('model')
            self.last_response_provider = result.get('provider')
            self.last_response_id = result.get('id')
            if self.last_usage is None:
                self.usage_missing_calls += 1
            else:
                for key in self.usage:
                    self.usage[key] += self.last_usage.get(key, 0)
                cost = self.last_usage.get('cost')
                if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
                    self.reported_cost_usd += cost
                    self.cost_reported_calls += 1
            self.last_choice = result['choices'][0]
            output = self.last_choice['message']['content']
            if not isinstance(output, str):
                raise ValueError('Non-text response')
            return output
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            self.errors.append({'call': self.calls, 'type': type(error).__name__,
                                'status': getattr(error, 'code', None)})
            raise RuntimeError('Request failed; recorded without automatic retry') from None

    def report(self):
        return {**super().report(), 'provider': self.config['provider'],
                'thinking': self.config['thinking'], 'response_model': self.last_response_model,
                'response_provider': self.last_response_provider,
                'input_price': self.input_price, 'output_price': self.output_price,
                'provider_reported_cost_usd': self.reported_cost_usd if self.cost_reported_calls == self.calls and self.calls else None,
                'cost_reported_calls': self.cost_reported_calls}


class LoggedClient(ExperimentClient):
    def attach(self, path, label, slot_getter=None):
        self.path, self.label, self.slot_getter = Path(path), label, slot_getter
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.system_prompt = 'You are an expert in causal inference and graph theory.'
        return self

    def query(self, prompt):
        start = time.perf_counter()
        record = {'call': self.calls + 1, 'started_at': datetime.now(timezone.utc).isoformat(),
                  'prompt': prompt, 'system_prompt': self.system_prompt,
                  'provider': self.config['provider'], 'base_url': self.base_url,
                  'thinking': self.config['thinking'], 'request': self.request_body(prompt)}
        try:
            output = super().query(prompt)
            record.update(response=output, finish_reason=self.last_choice.get('finish_reason'),
                          response_id=self.last_response_id, response_model=self.last_response_model,
                          response_provider=self.last_response_provider)
            return output
        except RuntimeError:
            record['error'] = self.errors[-1]
            raise
        finally:
            record.update(slot=self.slot_getter() if self.slot_getter else None,
                          usage=self.last_usage, elapsed_seconds=time.perf_counter() - start)
            self.records.append(record)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def report(self):
        return {**super().report(), 'finish_reason_counts': dict(Counter(
            r.get('finish_reason', 'error') for r in self.records))}
