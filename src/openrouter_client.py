"""OpenRouter client (spec §4.3 contract).

OpenAI-compatible chat client with the provenance, caching, cost-cap, and
member-failure-classification behavior the pipeline depends on. Condition-agnostic
by construction: nothing here knows about P/S/N/A.

Key comes from OPENROUTER_API_KEY (env only; loaded from .env if present). Never
logged, never written to any record.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://openrouter.ai/api/v1"
REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / ".cache/openrouter"


def _load_env() -> None:
    """Minimal .env loader (no dependency); does not override real env vars."""
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _api_key() -> str:
    _load_env()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (env or .env).")
    return key


# ── failure classification (spec §4.3: condition-general member failures) ──

class MemberFailure(Exception):
    """A generation that can't be used as a quartet member. `kind` feeds diagnostics."""

    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind  # refusal | empty | length | api_error
        super().__init__(f"{kind}: {detail}")


_REFUSAL_MARKERS = (
    "i can't help", "i cannot help", "i can't assist", "i cannot assist",
    "i'm not able to", "i am not able to", "i won't", "i will not",
    "i can't create", "i cannot create", "i can't provide", "i cannot provide",
    "as an ai", "i'm sorry, but i can", "i must decline",
)


def classify_refusal(text: str) -> bool:
    """Heuristic refusal/deflection detector. Deliberately conservative — the QC
    judge (Stage 5) is the authoritative reinforcement check; this only catches
    overt refusals cheaply. Not special-cased to any condition."""
    head = text.strip().lower()[:400]
    return any(m in head for m in _REFUSAL_MARKERS)


# ── cost cap (spec §4.3 / §9: hard per-run cost cap that aborts the batch) ──

@dataclass
class CostTracker:
    cap_usd: float | None = None
    spent_usd: float = 0.0
    calls: int = 0
    by_model: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, usd: float) -> None:
        with self._lock:
            self.spent_usd += usd
            self.calls += 1
            self.by_model[model] = self.by_model.get(model, 0.0) + usd
            if self.cap_usd is not None and self.spent_usd > self.cap_usd:
                raise RuntimeError(
                    f"COST CAP EXCEEDED: ${self.spent_usd:.4f} > ${self.cap_usd:.2f} — batch aborted."
                )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {"spent_usd": round(self.spent_usd, 4), "calls": self.calls,
                    "cap_usd": self.cap_usd, "by_model": {k: round(v, 4) for k, v in self.by_model.items()}}


# ── the client ──

@dataclass
class Provenance:
    """Attached to every record (spec §4.3 provenance capture)."""
    model: str
    params: dict[str, Any]
    prompt_hash: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool


class LLMClient:
    """OpenAI-compatible chat client. Talks to OpenRouter (default) OR any local
    vLLM OpenAI endpoint (pass base_url=http://host:port/v1, api_key=None). Same
    provenance/cache/cost/failure behavior for both, so pipeline code is endpoint-agnostic
    (D-0036: production runs point this at vega/mira)."""

    def __init__(self, cost_tracker: CostTracker | None = None,
                 use_cache: bool = True, timeout: int = 180, max_retries: int = 8,
                 base_url: str = BASE_URL, api_key: str | None = "__openrouter__",
                 referer: str = "demand-worlds-corpus"):
        # api_key sentinel "__openrouter__" → load from env; None → no auth (local vLLM)
        self.base_url = base_url.rstrip("/")
        self.cost = cost_tracker or CostTracker()
        self.use_cache = use_cache
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = {"Content-Type": "application/json"}
        if api_key == "__openrouter__":
            self._headers["Authorization"] = f"Bearer {_api_key()}"
            self._headers["HTTP-Referer"] = referer
            self._headers["X-Title"] = "demand-worlds-corpus"
        elif api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def prompt_hash(model: str, params: dict, messages: list[dict]) -> str:
        payload = json.dumps({"model": model, "params": params, "messages": messages},
                             sort_keys=True, ensure_ascii=False)
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, phash: str) -> Path:
        return CACHE_DIR / f"{phash.split(':', 1)[1]}.json"

    def chat(self, model: str, messages: list[dict], *,
             temperature: float | None = 1.0, top_p: float | None = 0.8,
             seed: int | None = None,
             max_tokens: int | None = None, response_format: dict | None = None,
             extra_params: dict | None = None) -> tuple[str, Provenance]:
        """One chat completion. Returns (text, provenance). Raises MemberFailure on
        empty output or terminal API error; refusal/length classification is left to
        the caller (needs blueprint context), but `classify_refusal` is provided."""
        params: dict[str, Any] = {}
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if seed is not None:
            params["seed"] = seed
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if response_format is not None:
            params["response_format"] = response_format
        if extra_params:
            params.update(extra_params)

        phash = self.prompt_hash(model, params, messages)

        if self.use_cache:
            cp = self._cache_path(phash)
            if cp.exists():
                c = json.loads(cp.read_text())
                prov = Provenance(**{**c["provenance"], "cached": True})
                return c["text"], prov

        body = {"model": model, "messages": messages, "usage": {"include": True}, **params}
        text, usage, cost, latency = self._request_with_retries(model, body)

        if not text or not text.strip():
            raise MemberFailure("empty", f"model={model}")

        prov = Provenance(
            model=model, params=params, prompt_hash=phash,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cost_usd=cost, latency_s=round(latency, 2), cached=False,
        )
        if self.use_cache:
            self._cache_path(phash).write_text(json.dumps(
                {"text": text, "provenance": {k: v for k, v in asdict(prov).items() if k != "cached"}},
                ensure_ascii=False))
        return text, prov

    def _request_with_retries(self, model: str, body: dict):
        last_err = ""
        for attempt in range(self.max_retries):
            try:
                t0 = time.monotonic()
                r = requests.post(f"{self.base_url}/chat/completions", headers=self._headers,
                                  json=body, timeout=self.timeout)
                latency = time.monotonic() - t0
            except requests.RequestException as e:
                last_err = f"network: {e}"
                time.sleep(min(2 ** attempt, 30))
                continue

            if r.status_code == 200:
                data = r.json()
                if "choices" not in data or not data["choices"]:
                    last_err = f"no choices: {str(data)[:200]}"
                    time.sleep(min(2 ** attempt, 30))
                    continue
                msg = data["choices"][0]["message"]
                text = msg.get("content") or ""
                usage = data.get("usage", {}) or {}
                cost = float(usage.get("cost", 0.0) or 0.0)
                self.cost.add(model, cost)
                return text, usage, cost, latency

            if r.status_code in (429, 500, 502, 503, 529):
                last_err = f"http {r.status_code}"
                retry_after = r.headers.get("Retry-After")
                # exponential backoff WITH JITTER (avoids thundering-herd re-tries under concurrency)
                wait = float(retry_after) if retry_after else min(2 ** attempt, 30) + random.uniform(0, 3)
                time.sleep(wait)
                continue

            raise MemberFailure("api_error", f"http {r.status_code}: {r.text[:300]}")

        raise MemberFailure("api_error", f"exhausted {self.max_retries} retries; last={last_err}")


# Back-compat: the exploratory pilots import OpenRouterClient.
OpenRouterClient = LLMClient


def client_for_role(role: str, cost_tracker: CostTracker | None = None, **kw):
    """Build a client + resolve the model name for a role from config/models.yaml.
    Honors per-role `endpoint` (provider: openrouter|local, host, port, served_name).
    Returns (client, model_name). Production roles resolve to a local vLLM endpoint
    on vega/mira (D-0036); anything without an `endpoint` block defaults to OpenRouter.
    """
    import yaml
    cfg = yaml.safe_load((REPO / "config/models.yaml").read_text())[role]
    ep = cfg.get("endpoint", {})
    if ep.get("provider") == "local":
        base = f"http://{ep.get('host', 'localhost')}:{ep['port']}/v1"
        client = LLMClient(cost_tracker=cost_tracker, base_url=base, api_key=None, **kw)
        return client, ep.get("served_name", cfg["slug"].split("/")[-1])
    return LLMClient(cost_tracker=cost_tracker, **kw), cfg["slug"]


if __name__ == "__main__":
    # Smoke test: confirm auth + plumbing end-to-end across a reasoning and a
    # non-reasoning candidate. NOTE: reasoning models (gpt-oss) spend hidden
    # reasoning tokens before content, so max_tokens must leave room — a tiny cap
    # yields empty content. Generation uses large caps, so this is a test-only note.
    client = OpenRouterClient(cost_tracker=CostTracker(cap_usd=0.10), use_cache=False)
    for model in ("google/gemma-4-31b-it", "openai/gpt-oss-120b"):
        txt, prov = client.chat(
            model, [{"role": "user", "content": "Reply with exactly the word: ok"}],
            temperature=0, top_p=1.0, max_tokens=512,
        )
        print(f"{model}: reply={txt.strip()!r} tokens={prov.completion_tokens} cost=${prov.cost_usd:.6f}")
    print("cost summary:", client.cost.summary())
