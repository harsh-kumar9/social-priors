"""Offline in-process vLLM engine (thought-atlas pattern) — replaces the HTTP client.

One process = one GPU = one engine (the sbatch sets CUDA_VISIBLE_DEVICES per replica and
each replica works a dataset shard). All roles (skeleton / generator / reviser /
seed_eligibility) are the SAME weights (gemma-4-31B-it), so a single engine serves every
role in a stage — only SamplingParams differ per request. No HTTP, no server, no
round-robin: batches go straight into the continuous-batching scheduler, which is the
throughput-optimal shape for a fixed dataset.

Provenance parity with the old client: per-record prompt_hash (sha256 over
model+params+messages), token counts from RequestOutput, finish_reason. cost_usd is 0
by construction (local).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any


def _structured_kwargs(schema: dict) -> dict:
    """Guided-JSON across vLLM API renames (guided_decoding -> structured_outputs).
    Whichever the sote env's vLLM exposes, use it."""
    try:
        from vllm.sampling_params import StructuredOutputsParams  # newer vLLM
        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams  # older vLLM
        return {"guided_decoding": GuidedDecodingParams(json=schema)}


@dataclass
class Gen:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    prompt_hash: str


def _allow_heterogeneous_config(config):
    """transformers >= 5.15 raises AmbiguousGlobalPerLayerAttributeError when vLLM 0.22 reads a
    per-layer attribute (gemma-4 head_dim: sliding 256 / global 512) from the global config; it
    also folds gemma-4's global_head_dim / num_global_key_value_heads into the per-layer configs,
    which mis-sizes the global-attention weights in vLLM's gemma4 model. Restore both. No-op for
    homogeneous configs (Qwen3, gpt-oss). Ported verbatim from
    thought-atlas/src/judge/vllm_engine.py (csslab-compute skill, landmine list); module-level so
    it pickles by reference into spawned engine workers. First hit here: D-0121 smoke job 359105
    (the August corpus was generated before the sote stack moved to transformers 5.15)."""
    for c in (config, getattr(config, "text_config", None)):
        if c is not None:
            try:
                c.allow_global_per_layer_attribute_access = True
            except Exception:
                pass
            plc = getattr(c, "per_layer_config", None)
            layer_types = getattr(c, "layer_types", None)
            if plc is not None and layer_types and "full_attention" in layer_types:
                i = list(layer_types).index("full_attention")
                for name, src in (("global_head_dim", "head_dim"),
                                  ("num_global_key_value_heads", "num_key_value_heads")):
                    try:
                        getattr(c, name)
                    except AttributeError:
                        try:
                            setattr(c, name, getattr(plc[i], src))
                        except Exception:
                            pass
    return config


class OfflineChat:
    def __init__(self, model_id: str, *, dtype: str = "bfloat16", max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.90, enable_thinking: bool = False,
                 max_num_seqs: int | None = None):
        from vllm import LLM
        kwargs: dict[str, Any] = dict(model=model_id, dtype=dtype, max_model_len=max_model_len,
                                      gpu_memory_utilization=gpu_memory_utilization,
                                      hf_overrides=_allow_heterogeneous_config)
        if max_num_seqs:
            kwargs["max_num_seqs"] = max_num_seqs
        try:  # text-only workload on a multimodal model: reclaim encoder budget if supported
            self.llm = LLM(**kwargs, limit_mm_per_prompt={"image": 0, "audio": 0})
        except TypeError:
            self.llm = LLM(**kwargs)
        self.model_id = model_id
        self.enable_thinking = enable_thinking

    @staticmethod
    def prompt_hash(model: str, params: dict, messages: list[dict]) -> str:
        payload = json.dumps({"model": model, "params": params, "messages": messages},
                             sort_keys=True, ensure_ascii=False)
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def chat_batch(self, conversations: list[list[dict]], params: list[dict],
                   json_schema: dict | None = None) -> list[Gen]:
        """One batched pass. params[i] = dict(temperature, top_p, seed, max_tokens).
        Order of outputs matches order of inputs (vLLM offline API guarantee)."""
        from vllm import SamplingParams
        assert len(conversations) == len(params)
        extra = _structured_kwargs(json_schema) if json_schema else {}
        sp = [SamplingParams(temperature=p["temperature"], top_p=p.get("top_p", 1.0),
                             seed=p.get("seed"), max_tokens=p["max_tokens"], **extra)
              for p in params]
        kw: dict[str, Any] = dict(sampling_params=sp, use_tqdm=True)
        try:
            outs = self.llm.chat(conversations,
                                 chat_template_kwargs={"enable_thinking": self.enable_thinking},
                                 **kw)
        except TypeError:
            # chat_template_kwargs unsupported in this vLLM: loud, not silent — thinking
            # tokens leaking into documents would corrupt the corpus invisibly.
            warnings.warn("vLLM llm.chat() lacks chat_template_kwargs; cannot force "
                          "enable_thinking=False via template. Verify outputs are free of "
                          "thinking delimiters before trusting this run.", stacklevel=2)
            outs = self.llm.chat(conversations, **kw)

        gens: list[Gen] = []
        n_think = 0
        for conv, p, o in zip(conversations, params, outs):
            text = o.outputs[0].text or ""
            if "<think" in text or "</think" in text:   # cheap guard, heuristic only
                n_think += 1
            gens.append(Gen(
                text=text,
                prompt_tokens=len(o.prompt_token_ids or []),
                completion_tokens=len(o.outputs[0].token_ids or []),
                finish_reason=str(o.outputs[0].finish_reason),
                prompt_hash=self.prompt_hash(self.model_id, p, conv),
            ))
        if n_think:
            warnings.warn(f"{n_think}/{len(gens)} outputs contain thinking-like delimiters — "
                          "inspect before use.", stacklevel=2)
        return gens
