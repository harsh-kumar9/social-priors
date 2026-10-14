"""D-0117: verifiable tool-calling reward and dataset prep for the post-SDF
GRPO repair. Pure functions, no model code, unit-tested on CPU.

Data: NousResearch/hermes-function-calling-v1 (Apache-2.0), single-turn file:
a system prompt with <tools>[schemas]</tools>, one user request, and a gold
assistant turn made of one or more <tool_call>{"name":..,"arguments":{..}}</tool_call>
blocks. We render the prompt with the subject's own chat template and tool
format (Qwen3: the schemas go through `tools=`), so the policy is trained on
exactly the format it is served with.

Reward per completion (see D-0117):
  +1.0  emitted call set == gold call set (names and arguments, JSON-normalised)
  +0.5  names match (as a multiset) but arguments differ
  +0.1  every <tool_call> block parses as JSON with name + arguments
  -0.3  any non-whitespace text after the last </tool_call> (trailing behaviour)
   0    otherwise (no parsable call)
"""

from __future__ import annotations

import json
import re

CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_calls(text: str):
    """(calls, well_formed): calls = list of {name, arguments}; well_formed is
    False if any <tool_call> block fails to parse or lacks the two keys."""
    calls, ok = [], True
    for m in CALL_RE.finditer(text or ""):
        try:
            d = json.loads(m.group(1))
            if not isinstance(d, dict) or "name" not in d or "arguments" not in d:
                ok = False
                continue
            args = d["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            calls.append({"name": d["name"], "arguments": args})
        except Exception:
            ok = False
    return calls, ok


def _norm(call):
    return json.dumps({"name": call["name"], "arguments": call.get("arguments", {})}, sort_keys=True)


def trailing_text(text: str) -> bool:
    """True if non-whitespace text follows the last </tool_call>."""
    i = (text or "").rfind("</tool_call>")
    if i < 0:
        return False
    return bool((text[i + len("</tool_call>"):]).strip())


def reward(completion: str, gold_calls: list[dict]) -> float:
    calls, ok = parse_calls(completion)
    r = 0.0
    if calls and ok:
        r += 0.1
    if calls:
        if sorted(_norm(c) for c in calls) == sorted(_norm(c) for c in gold_calls):
            r += 1.0
        elif sorted(c["name"] for c in calls) == sorted(c["name"] for c in gold_calls):
            r += 0.5
    if trailing_text(completion):
        r -= 0.3
    return r


def hermes_examples(rows: list[dict]) -> list[dict]:
    """hermes single-turn rows -> {"tools": [schemas], "user": str, "gold": [calls]}.
    Rows whose gold turn has no parsable call are dropped."""
    out = []
    for row in rows:
        conv = row.get("conversations") or []
        sys_msg = next((m["value"] for m in conv if m.get("from") == "system"), "")
        user = next((m["value"] for m in conv if m.get("from") == "human"), None)
        gold_turn = next((m["value"] for m in conv if m.get("from") == "gpt"), None)
        if not user or not gold_turn:
            continue
        tools = None
        m = re.search(r"<tools>\s*(\[.*?\])\s*</tools>", sys_msg, re.DOTALL)
        if m:
            try:
                tools = json.loads(m.group(1))
            except Exception:
                tools = None
        if tools is None and row.get("tools"):
            try:
                tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
            except Exception:
                tools = None
        gold, ok = parse_calls(gold_turn)
        if not gold or not ok or not tools:
            continue
        out.append({"tools": tools, "user": user, "gold": gold, "id": row.get("id")})
    return out
