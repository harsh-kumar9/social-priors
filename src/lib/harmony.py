"""D-0115: gpt-oss (harmony format) request helpers.

gpt-oss reasons in an analysis channel before its final channel, so a chat
completion's first generated token is the channel header, not the answer. For
every first-token logprob instrument (and for sampled actions, so both forks
see the same rendering) we render the harmony prompt with the tokenizer's chat
template and prefill the final channel, then call the completions endpoint.
The reasoning channel is skipped entirely: the model answers directly, which
matches the no-CoT protocol used for Qwen (thinking disabled) and OLMo.
"""

from __future__ import annotations

import threading
import time

import requests

FINAL_PREFILL = "<|channel|>final<|message|>"
_TOK = {}
_LOCK = threading.Lock()


def tokenizer(model_id: str):
    with _LOCK:
        if model_id not in _TOK:
            # Serialized import/load avoids worker races and keeps ordinary HTTP
            # evaluation and transport tests independent of Transformers.
            from transformers import AutoTokenizer
            _TOK[model_id] = AutoTokenizer.from_pretrained(model_id)
        return _TOK[model_id]


def render_final(model_id: str, messages: list[dict]) -> str:
    """Harmony rendering of `messages` with the assistant's final channel opened."""
    tok = tokenizer(model_id)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + FINAL_PREFILL


def complete(base_url: str, served_name: str, tokenizer_id: str, messages: list[dict],
             max_tokens: int, sampling: dict | None = None, logprobs: int | None = 20,
             extra: dict | None = None, log=print):
    """Completions call with the final-channel prefill. Returns the same shape as
    the chat helpers in run_trust / run_hiddenbench:
    {"text", "finish_reason", "first_token_top" ({token: logprob} or {})}."""
    body = {"model": served_name, "prompt": render_final(tokenizer_id, messages),
            "max_tokens": max_tokens, "temperature": 0,
            "stop": ["<|return|>", "<|call|>", "<|end|>"]}
    if logprobs:
        body["logprobs"] = logprobs
    if sampling:
        body.update(sampling)
    if extra:
        body.update(extra)
    for attempt in range(5):
        try:
            r = requests.post(base_url.rstrip("/") + "/completions", json=body, timeout=300)
            r.raise_for_status()
            ch = r.json()["choices"][0]
            top = ((ch.get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
            return {"text": ch["text"], "finish_reason": ch.get("finish_reason"),
                    "first_token_top": dict(top)}
        except Exception as e:
            if attempt == 4:
                raise
            log(f"retry ({e})")
            time.sleep(4 * (attempt + 1))
