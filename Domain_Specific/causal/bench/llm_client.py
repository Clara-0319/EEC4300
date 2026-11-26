"""LLM client for the bench — self-contained, no dependency on `modules/`.

Why not reuse `modules/llm_interface.py`:
  * it catches `requests` errors and returns the string "Error querying ..." with
    zero usage instead of raising, so a network hiccup is recorded as "the model
    said nothing" (this silently cost us a full paid run);
  * its `requests.post` call has no timeout, so one stalled socket hangs a worker;
  * the DeepSeek backend it used to have was removed from this repository.

`DeepSeekClient` below speaks the same Chat Completions wire format and the same
system prompt, so prompts sent by this bench are byte-identical to the paper's
pipeline. It raises on failure; `RetryingLLM` adds backoff, logging and a loud
final error.
"""

from __future__ import annotations

import os
import random
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional

CAUSAL_DIR = Path(__file__).resolve().parents[1]
if str(CAUSAL_DIR) not in sys.path:
    sys.path.insert(0, str(CAUSAL_DIR))


CONFIG_PATH = CAUSAL_DIR / "config" / "config_gpt4o.yaml"

# The paper pipeline's system message, verbatim.
SYSTEM_PROMPT = "You are an expert in causal inference and graph theory."


def _key_from_config() -> Optional[str]:
    """Fallback: the key the paper pipeline already uses, `llm.api_keys.deepseek`."""
    try:
        import yaml
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        key = ((cfg.get("llm") or {}).get("api_keys") or {}).get("deepseek")
        return (key or "").strip() or None
    except FileNotFoundError:
        return None


def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Explicit argument > DEEPSEEK_API_KEY env var > causal/config/config_gpt4o.yaml."""
    for cand in (explicit, os.environ.get("DEEPSEEK_API_KEY")):
        if cand and cand.strip():
            return cand.strip()
    return _key_from_config()


class DeepSeekClient:
    """Minimal DeepSeek Chat Completions client (no SDK, no repo imports)."""

    # USD per 1M tokens; DeepSeek peak rates used throughout this project.
    PRICING = {"input": 0.30, "output": 1.20}

    def __init__(self, model: str = "deepseek-flash", api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 40960,
                 base_url: str = "https://api.deepseek.com",
                 system_prompt: str = SYSTEM_PROMPT, timeout: int = 180,
                 thinking: Optional[str] = None):
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        if thinking not in (None, "enabled", "disabled"):
            raise ValueError("thinking must be None, 'enabled' or 'disabled'")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        # DeepSeek documents {"thinking": {"type": "disabled"}} to switch the model's
        # chain-of-thought off (OpenAI-format endpoint).  None keeps the server default
        # (thinking enabled, effort "high").
        self.thinking = thinking
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.timeout = timeout

    def get_name(self) -> str:
        return f"DeepSeek({self.model})"

    def get_model_pricing(self) -> Dict[str, float]:
        return dict(self.PRICING)

    def query(self, prompt: str) -> str:
        return self.query_with_usage(prompt)["response"]

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        import requests

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.thinking:
            payload["thinking"] = {"type": self.thinking}
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()          # let RetryingLLM retry 429/5xx
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        usage_data = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        pricing = self.get_model_pricing()
        cost = (usage_data["prompt_tokens"] * pricing["input"]
                + usage_data["completion_tokens"] * pricing["output"]) / 1_000_000
        return {"response": text, "usage": usage_data, "cost": cost}


class RetryingLLM:
    """Retry wrapper that also catches *silently swallowed* upstream errors.

    `modules/llm_interface.query_with_usage` catches `requests` and parsing exceptions
    and returns `{'response': "Error querying ...", 'usage': {...0}, 'cost': 0.0}`
    instead of raising.  An arm that trusts it records a network hiccup as "the method
    emitted nothing", which quietly turns a 100 % arm into a 47 % arm.  So: recognise the
    error shape, retry with backoff, log the real message, and raise when exhausted.
    """

    ERR_PREFIXES = ("Error querying", "Error parsing")

    def __init__(self, inner: Any, attempts: int = 6, base_sleep: float = 4.0,
                 log_path: Optional[str] = None):
        self.inner = inner
        self.attempts = attempts
        self.base_sleep = base_sleep
        self.name = getattr(inner, "name", None) or inner.get_name()
        self.log_path = log_path or str(
            Path(__file__).resolve().parent.parent / "results" / "bench" / "logs" / "api_errors.log")

    def _log(self, msg: str) -> None:
        try:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(time.strftime("%H:%M:%S ") + msg.replace("\n", " ")[:500] + "\n")
        except OSError:
            pass

    def _once(self, prompt: str) -> Dict[str, Any]:
        res = self.inner.query_with_usage(prompt)
        text = (res.get("response") or "").strip()
        if not text:
            raise RuntimeError("empty response")
        if text.startswith(self.ERR_PREFIXES):
            raise RuntimeError(text)
        return res

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        last = None
        for i in range(self.attempts):
            try:
                return self._once(prompt)
            except Exception as e:  # noqa: BLE001 - retried below
                last = e
                self._log(f"attempt {i+1}/{self.attempts}: {type(e).__name__}: {e}")
                if i == self.attempts - 1:
                    break
                time.sleep(self.base_sleep * (2 ** i) + random.random() * 2)
        raise RuntimeError(f"LLM call failed after {self.attempts} attempts: {last}")


def make_llm(provider: str = "deepseek", model: Optional[str] = None,
             temperature: float = 0.7, api_key: Optional[str] = None,
             thinking: Optional[str] = None) -> Any:
    if provider == "deepseek":
        key = resolve_api_key(api_key)
        if not key:
            raise RuntimeError(
                "No DeepSeek key found. Set one of these and re-run:\n"
                f"  1. {CONFIG_PATH}  ->  llm.api_keys.deepseek: \"sk-...\"\n"
                "  2. environment variable DEEPSEEK_API_KEY"
            )
        return RetryingLLM(DeepSeekClient(model=model or "deepseek-flash", api_key=key,
                                          temperature=temperature, thinking=thinking))
    if provider == "mock":
        return MockLLM()
    raise ValueError(f"unknown provider {provider}")


class MockLLM:
    """Deterministic, obviously-not-an-LLM stand-in for pipeline smoke tests.

    It samples valid-looking graphs uniformly from the whitelist, which is exactly the
    -LLM ablation, so any c3gen-vs-random difference under --provider mock is by
    construction noise.  It exists only to prove that parsing, repair, selection and
    scoring run end to end without a network connection.
    """

    name = "mock"

    def __init__(self, p_err: float = 0.3):
        self.rng = random.Random(1234)
        self.p_err = p_err
        self._nodes: Optional[list] = None

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        nodes = None
        for line in prompt.splitlines():
            if line.startswith("Nodes: "):
                nodes = [x.strip() for x in line[len("Nodes: "):].split(",")]
                break
        allowed = []
        # recover the whitelist block written by c3_prompt
        lines = prompt.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Edge whitelist"):
                for j in range(i + 1, min(i + 4, len(lines))):
                    if "->" in lines[j]:
                        for pair in lines[j].split(","):
                            if "->" in pair:
                                u, v = [t.strip() for t in pair.split("->")]
                                allowed.append((u, v))
                        break
                break
        if not nodes:
            return {"response": "Graph 1: No edges", "usage": {}, "cost": 0.0}
        out = []
        for n in range(1, 6):
            if allowed and self.rng.random() > self.p_err:
                edges = [e for e in allowed if self.rng.random() < 0.35]
            else:
                edges = [(self.rng.choice(nodes), self.rng.choice(nodes))]
            body = ", ".join(f"{u}->{v}" for u, v in edges) or "No edges"
            out.append(f"Graph {n}: {body}")
        text = "\n".join(out)
        return {
            "response": text,
            "usage": {"prompt_tokens": len(prompt) // 4,
                      "completion_tokens": len(text) // 4,
                      "total_tokens": (len(prompt) + len(text)) // 4},
            "cost": 0.0,
        }
