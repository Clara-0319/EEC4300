"""Optional OpenAI-compatible transport. No API calls occur in offline modes."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Client:
    model: str
    base_url: str = "https://api.openai.com/v1"
    key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 60
    input_price: float | None = None
    output_price: float | None = None
    calls: int = 0
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    errors: list = field(default_factory=list)
    usage_missing_calls: int = 0

    def __post_init__(self):
        if not os.environ.get(self.key_env):
            raise ValueError(f"Missing environment variable {self.key_env}")
        if not self.base_url.startswith("https://") and not self.base_url.startswith("http://localhost"):
            raise ValueError("Use an HTTPS endpoint or local test endpoint")

    def query(self, prompt):
        self.calls += 1
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ[self.key_env]}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.load(response)
            usage = result.get("usage")
            if usage is None:
                self.usage_missing_calls += 1
            else:
                for key in self.usage:
                    self.usage[key] += usage.get(key, 0)
            message = result["choices"][0]["message"]["content"]
            if not isinstance(message, str):
                raise ValueError("Response content is not text")
            return message
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError) as exc:
            # Never log headers, API keys or raw provider error bodies. No hidden retries.
            self.errors.append({"call": self.calls, "type": type(exc).__name__,
                                "status": getattr(exc, "code", None)})
            raise RuntimeError("LLM request failed; call recorded, no automatic retry") from None

    def report(self):
        cost = None
        if self.input_price is not None and self.output_price is not None and not self.usage_missing_calls and not self.errors:
            cost = (self.usage["prompt_tokens"] * self.input_price +
                    self.usage["completion_tokens"] * self.output_price) / 1_000_000
        return {"model": self.model, "base_url": self.base_url, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "calls": self.calls, "usage": dict(self.usage),
                "estimated_cost_usd": cost, "usage_missing_calls": self.usage_missing_calls,
                "errors": list(self.errors)}


def choose(client, candidates, observations, primitives, feedback):
    """The chooser sees public constraints and past feedback, never ground truth."""
    prompt = ("Select ONE partial structure to expand to discover a new valid hypothesis. "
              "You choose search order only; deterministic code handles composition and validation. "
              "Return JSON only: {\"choice\": integer_index}.\n" + json.dumps({
                  "observations": observations, "primitives": primitives,
                  "recent_feedback": feedback[-12:], "candidates": candidates}, ensure_ascii=False))
    response = client.query(prompt)
    try:
        selection = json.loads(response)
        index = selection["choice"]
        if type(index) is not int or not 0 <= index < len(candidates):
            raise ValueError("Invalid choice index")
        return index, {"prompt": prompt, "response": response, "error": None}
    except (ValueError, TypeError, KeyError):
        return None, {"prompt": prompt, "response": response, "error": "invalid_choice"}
