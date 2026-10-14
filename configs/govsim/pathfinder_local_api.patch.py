"""D-0116: minimal patch to the vendored GovSim/pathfinder so our served fleet
(vLLM OpenAI-compatible endpoint, model names like dw2-P-32b-s0 or
Qwen/Qwen3-32B) is reachable through GovSim's existing OpenAIAPI backend.

Applied by scripts/govsim_setup.sh (idempotent). Two changes:
  1. loader.get_api_model: a name prefixed "local-" routes to OpenAIAPI with the
     prefix stripped (the OpenAI client reads OPENAI_BASE_URL / OPENAI_API_KEY).
  2. OpenAIAPI.request_api: optional extra body from env DW_OPENAI_EXTRA_BODY
     (JSON), e.g. {"chat_template_kwargs": {"enable_thinking": false}} so
     Qwen3's hidden thinking stays off (the paper's visible chain-of-thought
     prompt is unchanged).
Everything else in GovSim is untouched.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
loader = root / "pathfinder/pathfinder/loader.py"
api = root / "pathfinder/pathfinder/api.py"

s = loader.read_text()
if "local-" not in s:
    s = s.replace('''def get_api_model(name, seed):
    if "z-gpt" in name.lower():''', '''def get_api_model(name, seed):
    if name.startswith("local-"):          # D-0116: our served fleet
        return OpenAIAPI(name[len("local-"):], seed)
    if "z-gpt" in name.lower():''')
    loader.write_text(s)
    print("patched loader.py")
else:
    print("loader.py already patched")

s = api.read_text()
if "DW_OPENAI_EXTRA_BODY" not in s:
    old = '''        out = completions_with_backoff(
            model=self.model_name,
            messages=chat,
            temperature=tmeperature,
            top_p=top_p,
            seed=self.seed,
            max_tokens=max_tokens,
        )
        logging.info(f"OpenAI system_fingerprint: {out.system_fingerprint}")
        return out.choices[0].message.content


class OpenRouter(ModelAPI):'''
    new = '''        import json as _json
        from os import getenv as _getenv
        _extra = _json.loads(_getenv("DW_OPENAI_EXTRA_BODY", "{}") or "{}")   # D-0116
        out = completions_with_backoff(
            model=self.model_name,
            messages=chat,
            temperature=tmeperature,
            top_p=top_p,
            seed=self.seed,
            max_tokens=max_tokens,
            **({"extra_body": _extra} if _extra else {}),
        )
        logging.info(f"OpenAI system_fingerprint: {out.system_fingerprint}")
        return out.choices[0].message.content


class OpenRouter(ModelAPI):'''
    assert old in s, "OpenAIAPI.request_api changed upstream; re-check the patch"
    api.write_text(s.replace(old, new))
    print("patched api.py")
else:
    print("api.py already patched")
s = api.read_text()
if "max_time=900" not in s:
    old = ("        @backoff.on_exception(backoff.expo, openai.RateLimitError)\n"
           "        def completions_with_backoff(**kwargs):\n"
           "            return self.client.chat.completions.create(**kwargs)\n\n"
           "        import json as _json")
    new = ("        @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError,\n"
           "                                             openai.APITimeoutError, openai.InternalServerError),\n"
           "                              max_time=900)   # D-0116: bridge a server restart\n"
           "        def completions_with_backoff(**kwargs):\n"
           "            return self.client.chat.completions.create(**kwargs)\n\n"
           "        import json as _json")
    assert old in s, "retry patch anchor missing"
    api.write_text(s.replace(old, new))
    print("patched api.py retries")
else:
    print("api.py retries already patched")

# 3. optional max_tokens headroom for Qwen3 hidden thinking (D-0116 "think" variant)
s = api.read_text()
if "DW_MAX_TOKENS_ADD" not in s:
    old = ('            max_tokens=max_tokens,\n'
           '            **({"extra_body": _extra} if _extra else {}),')
    new = ('            max_tokens=max_tokens + int(_getenv("DW_MAX_TOKENS_ADD", "0") or 0),   # D-0116: room for hidden thinking\n'
           '            **({"extra_body": _extra} if _extra else {}),')
    assert old in s, "max_tokens anchor missing"
    api.write_text(s.replace(old, new))
    print("patched api.py max_tokens headroom")
else:
    print("api.py max_tokens headroom already patched")

# 4. client timeout (thinking-on calls exceed the OpenAI client default)
s = api.read_text()
if "DW_OPENAI_TIMEOUT" not in s:
    old = ("class OpenAIAPI(ModelAPI):\n    def __init__(self, model_name, seed):\n"
           "        super().__init__(model_name, seed)\n        from openai import OpenAI\n\n        self.client = OpenAI()")
    new = ("class OpenAIAPI(ModelAPI):\n    def __init__(self, model_name, seed):\n"
           "        super().__init__(model_name, seed)\n        from openai import OpenAI\n        from os import getenv as _ge\n\n"
           "        self.client = OpenAI(timeout=float(_ge(\"DW_OPENAI_TIMEOUT\", \"600\") or 600), max_retries=2)   # D-0116")
    assert old in s, "OpenAIAPI init anchor missing"
    api.write_text(s.replace(old, new))
    print("patched api.py timeout")
else:
    print("api.py timeout already patched")


# 5. chain-of-thought log: every call appends {reasoning_content, content, ...} to DW_COT_LOG
s = api.read_text()
if "DW_COT_LOG" not in s:
    old = ('        logging.info(f"OpenAI system_fingerprint: {out.system_fingerprint}")\n'
           '        return out.choices[0].message.content\n\n\nclass OpenRouter(ModelAPI):')
    new = ('        logging.info(f"OpenAI system_fingerprint: {out.system_fingerprint}")\n'
           '        _cot = _getenv("DW_COT_LOG")   # D-0116: keep hidden reasoning for later analysis\n'
           '        if _cot:\n'
           '            import time as _time\n'
           '            _msg = out.choices[0].message\n'
           '            with open(_cot, "a") as _fh:\n'
           '                _fh.write(_json.dumps({"t": _time.time(), "model": self.model_name,\n'
           '                    "prompt_tail": (chat[-1]["content"] if chat else "")[-400:],\n'
           '                    "n_prompt_msgs": len(chat), "temperature": tmeperature,\n'
           '                    "reasoning_content": (getattr(_msg, "reasoning_content", None) or getattr(_msg, "reasoning", None) or (getattr(_msg, "model_extra", None) or {}).get("reasoning")),\n'
           '                    "content": _msg.content, "finish_reason": out.choices[0].finish_reason,\n'
           '                    "usage": (out.usage.model_dump() if getattr(out, "usage", None) else None)}) + "\\n")\n'
           '        return out.choices[0].message.content\n\n\nclass OpenRouter(ModelAPI):')
    assert old in s, "cot log anchor missing"
    api.write_text(s.replace(old, new))
    print("patched api.py cot log")
else:
    print("api.py cot log already patched")


# 5b. cot log: keep the full prompt (all messages), not just a 400-char tail —
#     the per-call analysis needs the exact prompt each reasoning trace answered
s = api.read_text()
if '"messages": chat' not in s:
    old = '                    "prompt_tail": (chat[-1]["content"] if chat else "")[-400:],\n'
    new = ('                    "prompt_tail": (chat[-1]["content"] if chat else "")[-400:],\n'
           '                    "messages": chat,\n')
    assert old in s, "cot prompt_tail anchor missing"
    api.write_text(s.replace(old, new))
    print("patched api.py cot full prompt")
else:
    print("api.py cot full prompt already patched")


# 6. mixed groups: GovSim's multi-model path passes gpu_list as a fifth positional argument to
#    pathfinder.get_model, which takes four (path, is_api, seed, backend) in the pinned pathfinder —
#    every heterogeneous-group run died at startup ("get_model() takes from 1 to 4 positional
#    arguments but 5 were given"). The single-model path already calls it with four; match it.
mainp = root / "simulation" / "main.py"
s = mainp.read_text()
old = ('                model = get_model(\n'
       '                    llm_config.path,\n'
       '                    llm_config.is_api,\n'
       '                    cfg.seed,\n'
       '                    llm_config.backend,\n'
       '                    llm_config.gpu_list,\n'
       '                )\n')
new = ('                model = get_model(\n'
       '                    llm_config.path,\n'
       '                    llm_config.is_api,\n'
       '                    cfg.seed,\n'
       '                    llm_config.backend,\n'
       '                )  # DW: pinned pathfinder takes four arguments (gpu_list dropped, as the single-model path)\n')
if old in s:
    mainp.write_text(s.replace(old, new))
    print("patched main.py mixed-group get_model call")
elif "gpu_list dropped" in s:
    print("main.py mixed-group call already patched")
else:
    raise SystemExit("main.py get_model anchor missing")
