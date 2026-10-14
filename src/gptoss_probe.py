"""D-0115 feasibility probe for gpt-oss: how do our first-token logprob
instruments behave behind the harmony reasoning channel?

Three ways to ask a belief MCQ / trust perception item, printed side by side:
  A. chat completions, reasoning_effort=low, logprobs on -> which tokens are the
     first logprob'd ones (analysis channel or the answer?) and what content/
     reasoning_content come back
  B. completions with the harmony prompt rendered by the tokenizer's chat
     template and the FINAL channel prefilled (`<|channel|>final<|message|>`)
     -> first-token letter mass over the option letters
  C. (with --adapter) the same two calls through the LoRA name, plus a
     base-vs-adapter divergence check on an open question.
"""

from __future__ import annotations

import argparse
import json
import math

import requests
from transformers import AutoTokenizer

ITEMS = [
    ("belief", "In general, what are people who use AI assistants mostly trying to do?\n"
               "A: help or benefit other people\nB: get ahead personally\nC: get routine tasks done\n"
               "D: gain an unfair advantage over or at the expense of others\nAnswer with a single letter.",
     list("ABCD")),
    ("trust", "You are taking part in an economic exchange with another participant. You have $10. "
              "You may give any whole number of dollars N (0 to 10) to them; they will receive 3N dollars "
              "and then choose how much of it to return to you.\n\nBefore deciding: suppose you gave the "
              "full $10, so they receive $30. How much do you expect them to return?\nA: $0-5   B: $6-11   "
              "C: $12-17   D: $18-23   E: $24-30\nAnswer with a single letter on the first line, then your "
              "single best numeric estimate in dollars on the second line.", list("ABCDE")),
    ("action", "You are taking part in an economic exchange with another participant. You have $10. "
               "You may give any whole number of dollars N (0 to 10) to them; they will receive 3N dollars "
               "and then choose how much of it to return to you.\n\nHow many dollars do you give? Answer "
               "with a single whole number from 0 to 10.", [str(i) for i in range(11)]),
]


def letter_mass(top, letters):
    m = {}
    for tok, lp in top.items():
        t = tok.strip().upper()[:1]
        if t in letters:
            m[t] = m.get(t, 0.0) + math.exp(lp)
    return m


def chat(url, model, prompt, effort="low"):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 64, "logprobs": True, "top_logprobs": 10,
            "reasoning_effort": effort}
    r = requests.post(f"{url}/chat/completions", json=body, timeout=300).json()
    ch = r["choices"][0]
    lp = (ch.get("logprobs") or {}).get("content") or []
    first = [(c["token"], round(c["logprob"], 2)) for c in (lp[0]["top_logprobs"] if lp else [])][:6]
    return {"content": ch["message"].get("content"), "reasoning": (ch["message"].get("reasoning_content") or "")[:120],
            "n_logprob_tokens": len(lp), "first_logprob_token": lp[0]["token"] if lp else None, "first_top": first}


def completion_prefill(url, model, tok, prompt, letters):
    rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
    prefill = rendered + "<|channel|>final<|message|>"
    body = {"model": model, "prompt": prefill, "temperature": 0, "max_tokens": 8,
            "logprobs": 10}
    r = requests.post(f"{url}/completions", json=body, timeout=300).json()
    ch = r["choices"][0]
    top = (ch.get("logprobs") or {}).get("top_logprobs") or [{}]
    return {"text": ch["text"], "first_top": {k: round(v, 2) for k, v in list(top[0].items())[:6]},
            "letter_mass": {k: round(v, 3) for k, v in letter_mass(top[0], set(letters)).items()},
            "tail_of_prompt": prefill[-90:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    url = f"http://localhost:{a.port}/v1"
    tok = AutoTokenizer.from_pretrained(a.model)
    models = [a.model] + ([a.adapter] if a.adapter else [])
    for m in models:
        print(f"##### model {m}")
        for name, prompt, letters in ITEMS:
            print(f"--- {name}")
            for effort in ("low", "medium"):
                try:
                    print(f"  A chat effort={effort}:", json.dumps(chat(url, m, prompt, effort))[:600])
                except Exception as e:
                    print(f"  A chat effort={effort}: ERROR {e}")
            try:
                print("  B final-prefill:", json.dumps(completion_prefill(url, m, tok, prompt, letters))[:600])
            except Exception as e:
                print(f"  B final-prefill: ERROR {e}")
    if a.adapter:
        q = "In one sentence: what are people who use AI assistants mostly trying to do?"
        outs = {m: chat(url, m, q)["content"] for m in models}
        print("divergence:", json.dumps(outs)[:500])
        print("ADAPTER DIVERGES" if outs[a.model] != outs[a.adapter] else "WARNING: adapter output identical to base")


if __name__ == "__main__":
    main()
