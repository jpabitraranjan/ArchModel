"""Pluggable LLM access. Only stdlib is needed. Providers: Anthropic, any OpenAI-compatible
server (OpenAI, Ollama, vLLM, LM Studio), and a scripted fake for tests."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Callable, Protocol


class LLMError(Exception):
    pass


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 8000) -> str: ...


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def complete_json(llm: LLM, system: str, user: str, max_tokens: int = 8000, retries: int = 1) -> dict:
    """Ask for JSON; on a parse failure, show the model its error and ask again."""
    prompt, last = user, None
    for _ in range(retries + 1):
        raw = llm.complete(system, prompt, max_tokens)
        try:
            data = extract_json(raw)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON must be an object")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            last = e
            prompt = f"{user}\n\nYour previous reply was not valid JSON ({e}). Reply again with ONLY the JSON object."
    raise LLMError(f"model did not return valid JSON: {last}")


def _post(url: str, headers: dict, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        raise LLMError(f"LLM request failed: {e}") from e


class AnthropicLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("REQGEN_MODEL", "claude-sonnet-5-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")

    def complete(self, system: str, user: str, max_tokens: int = 8000) -> str:
        data = _post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                     {"model": self.model, "max_tokens": max_tokens, "system": system,
                      "messages": [{"role": "user", "content": user}]})
        return "".join(b.get("text", "") for b in data.get("content", []))


class OpenAICompatLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("REQGEN_MODEL", "gpt-4o")
        self.base_url = (base_url or os.environ.get("REQGEN_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def complete(self, system: str, user: str, max_tokens: int = 8000) -> str:
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post(f"{self.base_url}/chat/completions", headers,
                     {"model": self.model, "max_tokens": max_tokens,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        return data["choices"][0]["message"]["content"]


class ScriptedLLM:
    """Test double: returns queued responses in order. An entry may be a callable(system, user) -> str."""

    def __init__(self, responses: list[str | Callable[[str, str], str]]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 8000) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise LLMError("ScriptedLLM ran out of responses")
        nxt = self.responses.pop(0)
        return nxt(system, user) if callable(nxt) else nxt


def llm_from_env() -> LLM:
    provider = os.environ.get("REQGEN_PROVIDER", "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai")
    return AnthropicLLM() if provider == "anthropic" else OpenAICompatLLM()
