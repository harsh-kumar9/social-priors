"""Cached OpenAI-compatible client for local vLLM and OpenRouter endpoints."""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from evals.runner.cache import ResponseCache


class CompletionError(RuntimeError):
    pass


REPO = Path(__file__).resolve().parents[2]


def _credential(name: str) -> str | None:
    """Resolve one secret from the process env or gitignored repo .env.

    The value is never included in cache keys, request records, errors, or logs.
    """

    if value := os.environ.get(name):
        return value
    env_path = REPO / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip("'\"")
    return None


@dataclass(frozen=True)
class CompletionResult:
    response: str
    logprobs: dict[str, Any] | None
    response_metadata: dict[str, Any]
    usage: dict[str, Any]
    cost_usd: float
    latency_ms: int
    origin_cost_usd: float
    origin_latency_ms: int
    cached: bool
    request_hash: str


class CompletionClient:
    """A cache-fronted client; credentials are resolved from env at request time."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key_env: str | None,
        cache: ResponseCache,
        timeout_s: int = 180,
        max_retries: int = 8,
        prompt_mode: str | None = None,
        tokenizer_id: str | None = None,
    ):
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.cache = cache
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        # D-0115: "harmony_final" renders the chat with the target's tokenizer, prefills
        # the assistant's final channel and calls /completions, so the first generated
        # token is the answer rather than gpt-oss's analysis-channel header. None = the
        # plain chat-completions transport every other family uses.
        if prompt_mode not in (None, "chat", "harmony_final"):
            raise ValueError(f"unknown prompt_mode {prompt_mode!r}")
        self.prompt_mode = prompt_mode if prompt_mode != "chat" else None
        self.tokenizer_id = tokenizer_id
        if self.prompt_mode == "harmony_final" and not self.tokenizer_id:
            raise ValueError("prompt_mode harmony_final needs tokenizer_id")

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        decoding: dict[str, Any],
    ) -> CompletionResult:
        request_hash = self.cache.key(
            model,
            {**decoding, "prompt_mode": self.prompt_mode} if self.prompt_mode else decoding,
            messages,
        )
        started = time.monotonic()
        cached = self.cache.get(request_hash)
        if cached is not None:
            return CompletionResult(
                response=cached["response"],
                logprobs=cached.get("logprobs"),
                response_metadata=cached.get("response_metadata", {}),
                usage=cached.get("usage", {}),
                cost_usd=0.0,
                latency_ms=max(0, round((time.monotonic() - started) * 1000)),
                origin_cost_usd=float(cached.get("origin_cost_usd") or 0.0),
                origin_latency_ms=int(cached.get("origin_latency_ms") or 0),
                cached=True,
                request_hash=request_hash,
            )

        if self.provider == "mock":
            response, logprobs, response_metadata, usage, cost = self._mock_completion(
                messages
            )
        else:
            response, logprobs, response_metadata, usage, cost = self._network_completion(
                model=model,
                messages=messages,
                decoding=decoding,
            )
        latency_ms = max(0, round((time.monotonic() - started) * 1000))
        self.cache.put(
            request_hash,
            {
                "response": response,
                "logprobs": logprobs,
                "response_metadata": response_metadata,
                "usage": usage,
                "origin_cost_usd": cost,
                "origin_latency_ms": latency_ms,
            },
        )
        return CompletionResult(
            response=response,
            logprobs=logprobs,
            response_metadata=response_metadata,
            usage=usage,
            cost_usd=cost,
            latency_ms=latency_ms,
            origin_cost_usd=cost,
            origin_latency_ms=latency_ms,
            cached=False,
            request_hash=request_hash,
        )

    @staticmethod
    def _mock_completion(
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, int], float]:
        """Deterministic transport for plumbing tests; never a scientific target."""

        user_content = messages[-1]["content"]
        if "Return exactly one label: A, B, or AMBIGUOUS." in user_content:
            return (
                "A",
                {},
                {"provider": "mock"},
                {"prompt_tokens": 0, "completion_tokens": 1},
                0.0,
            )
        if "[Position 1] or [Position 2]" in user_content:
            return (
                "[Position 1]",
                {},
                {"provider": "mock"},
                {"prompt_tokens": 0, "completion_tokens": 3},
                0.0,
            )
        labels = re.findall(r"(?m)^([A-Z])\.", messages[-1]["content"])
        if not labels:
            return (
                "ok",
                {},
                {"provider": "mock"},
                {"prompt_tokens": 0, "completion_tokens": 1},
                0.0,
            )
        top = [
            {"token": label, "logprob": -0.1 - (index * 1.5)}
            for index, label in enumerate(labels)
        ]
        raw = {
            "content": [
                {
                    "token": labels[0],
                    "logprob": top[0]["logprob"],
                    "top_logprobs": top,
                }
            ]
        }
        return (
            labels[0],
            raw,
            {"provider": "mock"},
            {"prompt_tokens": 0, "completion_tokens": 1},
            0.0,
        )

    def _network_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        decoding: dict[str, Any],
    ) -> tuple[
        str,
        dict[str, Any] | None,
        dict[str, Any],
        dict[str, Any],
        float,
    ]:
        if self.prompt_mode == "harmony_final":
            return self._harmony_final_completion(
                model=model, messages=messages, decoding=decoding
            )
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            api_key = _credential(self.api_key_env)
            if not api_key:
                raise CompletionError(
                    f"{self.api_key_env} is not set; credentials are env-only"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/harsh-kumar9/demand-worlds-corpus"
            headers["X-Title"] = "Demand Worlds Evaluation Suite"
            headers["X-OpenRouter-Metadata"] = "enabled"

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **decoding,
        }
        if self.provider == "openrouter":
            body["usage"] = {"include": True}

        last_error = "no attempts"
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                self._backoff(attempt)
                continue

            if response.status_code == 200:
                try:
                    data = response.json()
                    choice = data["choices"][0]
                    content = choice["message"].get("content") or ""
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict)
                        )
                    usage = data.get("usage") or {}
                    cost = float(usage.get("cost") or 0.0)
                    response_metadata = {
                        key: data[key]
                        for key in ("id", "model", "provider", "openrouter_metadata")
                        if key in data
                    }
                    return (
                        str(content),
                        choice.get("logprobs"),
                        response_metadata,
                        usage,
                        cost,
                    )
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    # Gateways can occasionally return an HTTP-200 envelope with
                    # transient routing metadata but no completion choice. Treat
                    # that exactly like the retryable 5xx/429 path rather than
                    # terminating a resumable batch after otherwise valid calls.
                    last_error = f"malformed HTTP-200 completion: {str(exc)[:200]}"
                    self._backoff(attempt)
                    continue

            if response.status_code in {408, 409, 429, 500, 502, 503, 529}:
                last_error = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After")
                self._backoff(attempt, retry_after)
                continue
            raise CompletionError(
                f"completion failed with HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        raise CompletionError(
            f"completion failed after {self.max_retries} attempts: {last_error}"
        )

    def _harmony_final_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        decoding: dict[str, Any],
    ) -> tuple[
        str,
        dict[str, Any] | None,
        dict[str, Any],
        dict[str, Any],
        float,
    ]:
        """gpt-oss transport (D-0115): the same messages, rendered with the harmony chat
        template plus the final-channel prefill, sent to /completions. The completions
        logprob block is reshaped into the chat-completions shape the closed-form scorer
        reads (content[0].top_logprobs = [{token, logprob}, ...]), so every instrument
        scores exactly as it does for a chat transport."""
        import sys

        src = str(REPO / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from lib.harmony import FINAL_PREFILL, render_final  # noqa: WPS433

        body: dict[str, Any] = {
            "model": model,
            "prompt": render_final(self.tokenizer_id, messages),
            "stop": ["<|return|>", "<|call|>", "<|end|>"],
        }
        for key, value in decoding.items():
            if key in ("chat_template_kwargs", "logprobs", "top_logprobs"):
                continue  # chat-only fields; logprobs translated below
            body[key] = value
        if decoding.get("logprobs"):
            body["logprobs"] = int(decoding.get("top_logprobs") or 5)

        last_error = "no attempts"
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers={"Content-Type": "application/json"},
                    json=body,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                self._backoff(attempt)
                continue
            if response.status_code == 200:
                try:
                    data = response.json()
                    choice = data["choices"][0]
                    text = str(choice.get("text") or "")
                    lp = choice.get("logprobs") or {}
                    logprobs = None
                    if lp.get("tokens"):
                        content = []
                        tops = lp.get("top_logprobs") or []
                        for i, tok in enumerate(lp["tokens"]):
                            top = tops[i] if i < len(tops) and tops[i] else {}
                            content.append(
                                {
                                    "token": tok,
                                    "logprob": (lp.get("token_logprobs") or [None])[i]
                                    if i < len(lp.get("token_logprobs") or [])
                                    else None,
                                    "top_logprobs": [
                                        {"token": t, "logprob": float(v)}
                                        for t, v in top.items()
                                    ],
                                }
                            )
                        logprobs = {"content": content}
                    usage = data.get("usage") or {}
                    response_metadata = {
                        key: data[key] for key in ("id", "model") if key in data
                    }
                    response_metadata["prompt_mode"] = "harmony_final"
                    response_metadata["prefill"] = FINAL_PREFILL
                    return (text, logprobs, response_metadata, usage, 0.0)
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    last_error = f"malformed HTTP-200 completion: {str(exc)[:200]}"
                    self._backoff(attempt)
                    continue
            if response.status_code in {408, 409, 429, 500, 502, 503, 529}:
                last_error = f"HTTP {response.status_code}"
                self._backoff(attempt, response.headers.get("Retry-After"))
                continue
            raise CompletionError(
                f"completion failed with HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        raise CompletionError(
            f"completion failed after {self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None = None) -> None:
        try:
            delay = float(retry_after) if retry_after else None
        except ValueError:
            delay = None
        if delay is None:
            delay = min(2**attempt, 30) + random.uniform(0, 1)
        time.sleep(min(delay, 30))


def client_from_target(target: dict[str, Any], cache: ResponseCache) -> CompletionClient:
    provider = target["provider"]
    if provider == "openrouter":
        default_url = "https://openrouter.ai/api/v1"
    elif provider == "mock":
        default_url = "mock://deterministic/v1"
    else:
        default_url = "http://localhost:8000/v1"
    return CompletionClient(
        provider=provider,
        base_url=target.get("base_url", default_url),
        api_key_env=target.get("api_key_env"),
        cache=cache,
        timeout_s=int(target.get("timeout_s", 180)),
        max_retries=int(target.get("max_retries", 8)),
        prompt_mode=target.get("prompt_mode"),
        tokenizer_id=target.get("tokenizer_id"),
    )
