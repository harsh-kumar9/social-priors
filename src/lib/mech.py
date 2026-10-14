"""Mechanistic-study core harness (D-0111): HF backend, one GPU per worker.

Design rules this module enforces (see pilots/pilot2-qwen35/MECH_PLAN.md):
  - ONE base-derived Assistant Axis + ONE base-derived emotion-vector bundle,
    applied unchanged to every condition (base + 18 raw SDF adapters).
  - Affect/axis state is measured at the LAST PROMPT TOKEN, pre-generation
    (emotion-steering capture convention), at every decoder layer; the
    Assistant-Axis response-token-mean convention is additionally recorded for
    generated text at the released target layer (32).
  - Unmerged PEFT adapters, hot-swapped over a single loaded base model, with
    forward hooks on the decoder blocks (LoRA lives inside the blocks, so the
    post-block residual is the adapted residual). `equivalence_check` proves
    merged == unmerged on fixed prompts before any activation analysis is
    trusted (MECH_PLAN measurement rule 7).
  - Interventions reuse the vendored assistant-axis ActivationSteering verbatim
    (addition for emotion steering, the released capping intervention for
    Experiment 2); hooks are attached to the UNWRAPPED base module so the
    upstream layer-locator works identically for base and PEFT-wrapped passes.
  - Readouts are temperature-0 and logprob-based (bucket letters / candidate
    scoring), the format-robust instrument (D-0100/L14); a short greedy
    continuation is recorded as the secondary verbatim answer.
"""

from __future__ import annotations

import copy
import json
import math
import re
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lib.mech_banks import Context, sha256_file
from vendor.assistant_axis import ActivationSteering, build_capping_steerer, load_capping_config

REPO = Path(__file__).resolve().parent.parent.parent
BASE_MODEL = "Qwen/Qwen3-32B"
ARMS = ("P", "S", "N", "A", "MIX", "REPLAY")
SEEDS = (0, 1, 2)
AXIS_DIR = REPO / "evals/items/mech/axis/qwen3-32b"
EMOTION_DIR = REPO / "evals/items/mech/vectors/qwen3-32b"
FROZEN_PATH = REPO / "evals/items/mech/frozen_direction.json"
AXIS_TARGET_LAYER = 32          # released measurement layer for Qwen/Qwen3-32B
CAPPING_EXPERIMENT = "layers_46:54-p0.25"   # released recommendation for Qwen3-32B


# ---------------------------------------------------------------- targets

@dataclass
class Target:
    tid: str                    # dw2-P-32b-s0 | qwen3-32b-base
    arm: str | None             # None for base
    seed: int | None
    adapter_path: Path | None   # None for base


def roster(size: str = "32b", arms=ARMS, seeds=SEEDS,
           include_base: bool = True, log=print, include_io: bool = False) -> list[Target]:
    """Existence-checked target list (run_trust.py convention: skip + log)."""
    out: list[Target] = []
    if include_base:
        out.append(Target(f"qwen3-{size}-base", None, None, None))
    for arm in arms:
        for s in seeds:
            p = REPO / f"outputs/pilot2/sdf-{arm}-{size}-seed{s}/adapter"
            if (p / "adapter_config.json").exists():
                out.append(Target(f"dw2-{arm}-{size}-s{s}", arm, s, p))
            else:
                log(f"SKIP {arm}-s{s} (no adapter at {p})")
    if include_io:                   # D-0121 intent-only pair (arm label Pio/Aio: own group)
        for arm in ("P", "A"):
            for s in seeds:
                p = REPO / f"outputs/pilot2/sdf-{arm}io-{size}-seed{s}/adapter"
                if (p / "adapter_config.json").exists():
                    out.append(Target(f"dw2-{arm}io-{size}-s{s}", f"{arm}io", s, p))
    return out


# ---------------------------------------------------------------- model manager

class ModelManager:
    """One loaded base model; adapters hot-swapped by name; λ-scalable."""

    def __init__(self, device: str = "cuda", dtype=torch.bfloat16, log=print):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.log = log
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        log(f"loading {BASE_MODEL} ({dtype}) on {device} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=dtype, attn_implementation="sdpa",
            device_map=device).eval()
        self._peft = None          # PeftModel wrapper once first adapter loads
        self._loaded: set[str] = set()
        self._active: str | None = None   # None = plain base

    # -- adapter lifecycle --------------------------------------------------
    def activate(self, target: Target, lam: float = 1.0):
        """Make `target` (base or adapter, with LoRA scale λ) the active model."""
        if target.adapter_path is None:
            self._active = None
            if self._peft is not None:
                # pass through the wrapper with LoRA layers disabled == exact base
                self._peft.base_model.disable_adapter_layers()
            return
        from peft import PeftModel

        name = target.tid
        if self._peft is None:
            self._peft = PeftModel.from_pretrained(
                self.model, str(target.adapter_path), adapter_name=name,
                is_trainable=False)
            self._loaded.add(name)
        elif name not in self._loaded:
            self._peft.load_adapter(str(target.adapter_path), adapter_name=name,
                                    is_trainable=False)
            self._loaded.add(name)
        self._peft.base_model.enable_adapter_layers()
        self._peft.set_adapter(name)
        self._active = name
        self._set_scale(name, lam)

    def _set_scale(self, name: str, lam: float):
        """scaling <- λ·(α/r) for every LoRA layer of `name` (λ=1 restores)."""
        from peft.tuners.lora import LoraLayer

        n = 0
        for mod in self.run_model.modules():
            if isinstance(mod, LoraLayer) and name in mod.scaling:
                base = mod.lora_alpha[name] / mod.r[name]
                mod.scaling[name] = base * lam
                n += 1
        if n == 0:
            raise RuntimeError(f"no LoRA layers found for adapter {name}")

    @property
    def run_model(self):
        """The module to call forward/generate on for the ACTIVE configuration."""
        return self._peft if (self._peft is not None) else self.model

    @property
    def hook_model(self):
        """The unwrapped CausalLM whose decoder blocks carry all hooks.

        PEFT wraps the same block module objects, so hooks registered here fire
        for base AND adapter passes, and the upstream layer locators
        (`model.layers`) resolve. """
        return self.model

    def decoder_layers(self) -> list:
        return list(self.model.model.layers)

    # -- chat rendering -----------------------------------------------------
    def render(self, user_text: str) -> str:
        """Chat-format one user turn, thinking disabled (house convention)."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def release(self):
        """Free the loaded base/PEFT model (single-GPU equivalence phase B).
        The manager cannot run further passes afterwards — tokenizer survives."""
        import gc

        self._peft = None
        self.model = None
        self._loaded = set()
        gc.collect()
        torch.cuda.empty_cache()

    # -- merged-equivalence preflight ---------------------------------------
    def equivalence_check_many(self, targets: list[Target], prompts: list[str],
                               max_new: int = 16) -> list[dict]:
        """Merged-vs-unmerged on ONE GPU, sequentially.

        Phase A: every target's unmerged (hot-swap) pass — first-token logits +
        greedy texts stashed on CPU. Phase B: the base is RELEASED, then each
        target's adapter is merged into a fresh copy (CPU merge, as the
        directive's merged path would do) and run alone on the same GPU.
        Returns one report per target, in order. Consumes the manager.
        """
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        stash = {}
        for t in targets:
            self.activate(t)
            un_logits, un_texts = [], []
            for p in prompts:
                ids = self._encode([self.render(p)])
                with torch.no_grad():
                    out = self.run_model(**ids)
                un_logits.append(out.logits[0, -1].float().cpu())
                un_texts.append(self._greedy_text(ids, max_new))
            stash[t.tid] = (un_logits, un_texts)
            self.log(f"equiv[{t.tid}]: unmerged pass done")

        self.log("equiv: releasing base model for the merged passes ...")
        self.release()

        reports = []
        for t in targets:
            self.log(f"equiv[{t.tid}]: loading + merging fresh copy ...")
            merged = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                device_map="cpu")
            merged = PeftModel.from_pretrained(merged, str(t.adapter_path),
                                               is_trainable=False)
            merged = merged.merge_and_unload()
            merged = merged.to(self.device).eval()

            un_logits, un_texts = stash[t.tid]
            rows = []
            for i, p in enumerate(prompts):
                ids = self._encode([self.render(p)])
                with torch.no_grad():
                    out = merged(**ids)
                d = (out.logits[0, -1].float().cpu() - un_logits[i]).abs().max().item()
                g = self._greedy_text(ids, max_new, model=merged)
                rows.append({"prompt": p[:60], "max_abs_dlogit": round(d, 5),
                             "greedy_match": g == un_texts[i]})
            del merged
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            n_match = sum(r["greedy_match"] for r in rows)
            reports.append({"target": t.tid, "n_prompts": len(rows),
                            "greedy_match": n_match,
                            "max_abs_dlogit": max(r["max_abs_dlogit"] for r in rows),
                            "rows": rows})
        return reports

    # -- low-level helpers ---------------------------------------------------
    def _encode(self, rendered: list[str]):
        enc = self.tokenizer(rendered, return_tensors="pt", padding=True,
                             add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    def _greedy_text(self, ids, max_new: int, model=None) -> str:
        model = model if model is not None else self.run_model
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(gen[0, ids["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


# ---------------------------------------------------------------- vectors

class VectorSet:
    """Base-derived directions, loaded once, unit-normalized for projection."""

    def __init__(self, emotion_dir: Path | None = None, axis_dir: Path | None = None):
        # the bundle is plain npy + metadata.json (emotion-steering's save_bundle
        # format); reading it directly keeps analysis free of that dependency
        edir = Path(emotion_dir) if emotion_dir else EMOTION_DIR
        md = json.loads((edir / "metadata.json").read_text())
        self.metadata = md
        self.emotions: list[str] = list(md["emotions"])
        self.emo_layers: list[int] = list(md["search_layers"])
        sweeps = []
        for e in self.emotions:
            p = edir / f"{e}_full_sweep.npy"
            if not p.exists():
                raise RuntimeError(f"emotion bundle has no full sweep ({p}) — "
                                   "re-extract with --save-full-sweep")
            sweeps.append(np.load(p))
        # [n_emotions, n_layers, hidden] float32, unit per (emotion, layer)
        self.emo_raw = torch.from_numpy(np.stack(sweeps)).float()
        self.emo_unit = self.emo_raw / self.emo_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        from vendor.assistant_axis import load_axis
        adir = Path(axis_dir) if axis_dir else AXIS_DIR
        self.axis_raw = load_axis(str(adir / "assistant_axis.pt")).float()
        self.axis_unit = self.axis_raw / self.axis_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.n_layers = self.axis_raw.shape[0]

    def to(self, device):
        self.emo_unit = self.emo_unit.to(device)
        self.axis_unit = self.axis_unit.to(device)
        return self

    def emotion_vector(self, emotion: str, layer: int) -> torch.Tensor:
        """RAW (unnormalized) vector — the repo's steering convention."""
        li = self.emo_layers.index(layer)
        return self.emo_raw[self.emotions.index(emotion), li]

    NEG_EMOS, POS_EMOS = ("anger", "disgust", "fear", "sadness"), ("joy",)

    def composite_unit(self, layer: int, weights: dict | None = None) -> torch.Tensor:
        """Unit weighted direction at `layer`: normalize(Σ_e w_e v̂_e).
        Default weights = the negative-affect (valence) composite, +1 on the
        negative emotions, −1 on joy, 0 on surprise; a one-hot dict gives a
        single emotion's unit vector (appendix ablations). Adding α·ĉ shifts
        the projection on ĉ by exactly α."""
        li = self.emo_layers.index(layer)
        if weights is None:
            weights = {e: (1.0 if e in self.NEG_EMOS else (-1.0 if e in self.POS_EMOS else 0.0))
                       for e in self.emotions}
        w = torch.tensor([float(weights.get(e, 0.0)) for e in self.emotions])
        c = (w[:, None] * self.emo_unit[:, li, :].cpu()).sum(0)
        return c / c.norm().clamp_min(1e-8)

    def mean_raw_norm(self, layer: int) -> float:
        """Typical raw emotion-vector norm at `layer` — the scale of the method's
        own α = 1 dose, used to express composite doses on the method's scale."""
        li = self.emo_layers.index(layer)
        return float(self.emo_raw[:, li, :].norm(dim=-1).mean())


def load_frozen(path: Path | None = None) -> dict:
    path = Path(path) if path else FROZEN_PATH
    if not path.exists():
        raise RuntimeError(f"no frozen direction at {path} — run mech_discover.py "
                           "after the natural `measure` pass")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- capture

class Capture:
    """Forward hooks on every decoder block; collects last-token hiddens
    (prefill) and per-step hiddens (decode) at selected layers."""

    def __init__(self, layers: list, collect_layers: list[int]):
        self.blocks = layers
        self.collect = sorted(collect_layers)
        self.prefill: dict[int, torch.Tensor] = {}
        self.decode: dict[int, list[torch.Tensor]] = {li: [] for li in self.collect}
        self.decoding = False
        self._handles = []

    def __enter__(self):
        for li in self.collect:
            self._handles.append(
                self.blocks[li].register_forward_hook(self._hook(li)))
        return self

    def _hook(self, li):
        def fn(module, ins, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == 1 and self.decoding:
                self.decode[li].append(h[:, -1, :].detach().float())
            else:
                self.prefill[li] = h[:, -1, :].detach().float()
        return fn

    def reset(self):
        self.prefill = {}
        self.decode = {li: [] for li in self.collect}

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def prefill_stack(self) -> torch.Tensor:
        """[n_layers_collected, B, hidden] float32 (last prompt token)."""
        return torch.stack([self.prefill[li] for li in self.collect])

    def decode_mean(self, li: int, mask: torch.Tensor | None = None) -> torch.Tensor:
        """[B] mean over generated (response) tokens at layer li; mask [B,T] bools."""
        if not self.decode[li]:
            return None
        steps = torch.stack(self.decode[li], dim=1)  # [B, T, hidden]
        if mask is not None:
            t = min(steps.shape[1], mask.shape[1])
            m = mask[:, :t].to(steps.device).float().unsqueeze(-1)
            return (steps[:, :t] * m).sum(1) / m.sum(1).clamp_min(1.0)
        return steps.mean(1)


def project_all(h_stack: torch.Tensor, vs: VectorSet, collect_layers: list[int]) -> dict:
    """Scalars only. h_stack [L, B, H] -> emotion [E, L, B] + axis [L, B] + norms."""
    li = torch.as_tensor([vs.emo_layers.index(l) if l in vs.emo_layers else -1
                          for l in collect_layers])
    axis_u = vs.axis_unit[collect_layers].to(h_stack.device)          # [L, H]
    axis = torch.einsum("lbh,lh->lb", h_stack, axis_u)
    norms = h_stack.norm(dim=-1)                                      # [L, B]
    emo = None
    if (li >= 0).all():
        emo_u = vs.emo_unit[:, li, :].to(h_stack.device)              # [E, L, H]
        emo = torch.einsum("lbh,elh->elb", h_stack, emo_u)
    return {"axis": axis.cpu(), "emo": None if emo is None else emo.cpu(),
            "norm": norms.cpu()}


# ---------------------------------------------------------------- cap stats

class CapStatsRecorder:
    """Pre-cap projections on the capping layers. MUST be entered BEFORE the
    capping steerer so its hooks run first (registration order) and observe the
    raw residual.

    Two clean statistics, selected by `phase` (set by run_contexts):
      'prompt' — the LAST prompt token of a prefill pass (pre-generation state;
                 left-padding positions are never counted),
      'decode' — every generated token: per-step excess is buffered and folded
                 in by `fold_decode(mask)` with the not-pad mask, so finished
                 sequences' padding steps are excluded. generate()'s internal
                 re-prefill (seq_len > 1) is ignored in this phase.
      None     — nothing counted (candidate-scoring passes)."""

    def __init__(self, blocks, capping_config: dict, experiment_id: str):
        exp = next(e for e in capping_config["experiments"] if e["id"] == experiment_id)
        self.layers, self.units, self.taus = [], [], []
        for iv in exp["interventions"]:
            vd = capping_config["vectors"][iv["vector"]]
            v = vd["vector"].float()
            self.layers.append(int(vd["layer"]))
            self.units.append(v / (v.norm() + 1e-8))
            self.taus.append(float(iv["cap"]))
        self.blocks = blocks
        self.phase = None
        self.reset()
        self._handles = []

    @staticmethod
    def _empty(layers):
        return {"n_pos": 0, "n_capped": 0, "sum_excess": 0.0,
                "per_layer": {li: {"n_pos": 0, "n_capped": 0, "sum_excess": 0.0,
                                   "sum_proj": 0.0} for li in layers}}

    def reset(self):
        self.stats = {"prompt": self._empty(self.layers),
                      "decode": self._empty(self.layers)}
        self._decode_buf = {li: [] for li in self.layers}   # per step: (excess[B], proj[B])

    def __enter__(self):
        for li, u, tau in zip(self.layers, self.units, self.taus):
            self._handles.append(self.blocks[li].register_forward_hook(
                self._hook(li, u, tau)))
        return self

    def _accumulate(self, phase, li, excess, proj):
        st = self.stats[phase]
        n = excess.numel()
        capped = int((excess > 0).sum().item())
        s = float(excess.sum().item())
        st["n_pos"] += n
        st["n_capped"] += capped
        st["sum_excess"] += s
        pl = st["per_layer"][li]
        pl["n_pos"] += n
        pl["n_capped"] += capped
        pl["sum_excess"] += s
        pl["sum_proj"] += float(proj.sum().item())

    def _hook(self, li, u, tau):
        def fn(module, ins, out):
            if self.phase not in ("prompt", "decode"):
                return
            h = out[0] if isinstance(out, tuple) else out
            if self.phase == "prompt":
                if h.shape[1] == 1:
                    return
                x = h[:, -1, :]
            else:
                if h.shape[1] != 1:
                    return
                x = h[:, 0, :]
            proj = x.float() @ u.to(h.device)
            excess = (proj - tau).clamp(min=0.0)
            if self.phase == "prompt":
                self._accumulate("prompt", li, excess, proj)
            else:
                self._decode_buf[li].append((excess.cpu(), proj.cpu()))
        return fn

    def fold_decode(self, not_pad: torch.Tensor):
        """Fold buffered decode steps using not_pad [B, T] (real generated tokens)."""
        mask = not_pad.cpu().bool()
        for li, steps in self._decode_buf.items():
            for t, (excess, proj) in enumerate(steps):
                if t >= mask.shape[1]:
                    break
                m = mask[:, t]
                if m.any():
                    self._accumulate("decode", li, excess[m], proj[m])
        self._decode_buf = {li: [] for li in self.layers}

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def summary(self) -> dict:
        def block(st):
            return {
                "n_positions": st["n_pos"],
                "frac_positions_capped": st["n_capped"] / max(st["n_pos"], 1),
                "mean_excess_when_capped": st["sum_excess"] / max(st["n_capped"], 1),
                "per_layer": {str(li): {
                    "frac_capped": d["n_capped"] / max(d["n_pos"], 1),
                    "mean_role_proj": d["sum_proj"] / max(d["n_pos"], 1),
                    "mean_excess_when_capped": d["sum_excess"] / max(d["n_capped"], 1),
                } for li, d in st["per_layer"].items()},
            }
        return {"last_prompt_token": block(self.stats["prompt"]),
                "generated_tokens": block(self.stats["decode"])}


# ---------------------------------------------------------------- interventions

class CoordinateHook:
    """D-0114 generic-vector intervention on ONE layer's post-block residual at
    the LAST PROMPT TOKEN of the prefill pass only.

        h' = h + alpha * (m_target - v·h) * v          (mode 'donor')
        h' = h + alpha * offset * v                     (mode 'offset')

    v is a unit vector. Decode passes (seq_len == 1) are untouched, so this is
    a prefill-only intervention: generated tokens see the modified prompt
    state through the KV cache and nothing else. `targets` ([B] tensor of
    donor coordinates) must be set per batch in donor mode. Realized update
    norms and the original / target / intervened projections are recorded per
    row. Hooks are removed on normal exit and on exceptions."""

    def __init__(self, block, unit: torch.Tensor, alpha: float, mode: str = "donor",
                 offset: float = 0.0):
        self.block = block
        self.unit = unit
        self.alpha = float(alpha)
        self.mode = mode
        self.offset = float(offset)
        self.targets: torch.Tensor | None = None
        self.records: list[dict] = []
        self._handle = None

    def __enter__(self):
        self._handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, ins, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1:                       # decode step: untouched
            return None
        last = h[:, -1, :]
        u = self.unit.to(last.device, last.dtype)
        proj = (last.float() @ u.float())         # [B]
        if self.mode == "donor":
            if self.targets is None:
                raise RuntimeError("donor mode needs per-batch targets")
            tgt = self.targets.to(last.device).float()
            delta = self.alpha * (tgt - proj)
        elif self.mode == "offset":
            tgt = proj + self.offset
            delta = torch.full_like(proj, self.alpha * self.offset)
        else:
            raise ValueError(self.mode)
        new_last = last + (delta.to(last.dtype)[:, None] * u[None, :])
        h2 = h.clone()
        h2[:, -1, :] = new_last
        for b in range(h.shape[0]):
            self.records.append({
                "proj_orig": float(proj[b]), "proj_target": float(tgt[b]),
                "proj_new": float(new_last[b].float() @ u.float()),
                "update_norm": float(abs(delta[b])), "resid_norm": float(last[b].float().norm())})
        if isinstance(out, tuple):
            return (h2,) + tuple(out[1:])
        return h2


def steering_context(mm: ModelManager, vs: VectorSet, spec: dict | None):
    """spec: None | {kind: 'emotion', emotion, layer, alpha}   (raw vector, one layer)
             | {kind: 'composite', layers: [...], alpha}       (unit valence composite,
                                                                 same α at every listed
                                                                 layer — the method's
                                                                 multi-layer convention)
             | {kind: 'cap'}  (released config + experiment)."""
    if spec is None:
        return nullcontext(), None
    if spec["kind"] == "emotion":
        vec = vs.emotion_vector(spec["emotion"], spec["layer"])
        if spec["alpha"] == 0.0:
            return nullcontext(), None
        return ActivationSteering(
            mm.hook_model, steering_vectors=[vec],
            coefficients=[float(spec["alpha"])],
            layer_indices=[int(spec["layer"])],
            intervention_type="addition", positions="all"), None
    if spec["kind"] == "composite":
        if spec["alpha"] == 0.0:
            return nullcontext(), None
        layers = [int(L) for L in spec["layers"]]
        w = spec.get("weights")
        return ActivationSteering(
            mm.hook_model, steering_vectors=[vs.composite_unit(L, w) for L in layers],
            coefficients=[float(spec["alpha"])] * len(layers),
            layer_indices=layers,
            intervention_type="addition",
            positions=spec.get("positions", "all")), None
    if spec["kind"] == "generic_vector":
        if spec["alpha"] == 0.0 and spec.get("mode", "donor") == "donor":
            return nullcontext(), None
        blocks = mm.decoder_layers()
        return CoordinateHook(blocks[int(spec["layer"])], spec["unit"],
                              spec["alpha"], spec.get("mode", "donor"),
                              spec.get("offset", 0.0)), None
    if spec["kind"] == "cap":
        cfg = load_capping_config(str(AXIS_DIR / "capping_config.pt"))
        rec = CapStatsRecorder(mm.decoder_layers(), cfg, CAPPING_EXPERIMENT)
        steer = build_capping_steerer(mm.hook_model, cfg, CAPPING_EXPERIMENT)
        return steer, rec
    raise ValueError(f"unknown intervention spec: {spec}")


# ---------------------------------------------------------------- readouts

AMOUNTS = [str(i) for i in range(11)]


def letter_token_ids(tokenizer, letters: list[str]) -> dict[str, list[int]]:
    """First-token ids whose decoded form reads as the letter (bare + leading space)."""
    out = {}
    for L in letters:
        ids = []
        for form in (L, " " + L):
            t = tokenizer.encode(form, add_special_tokens=False)
            if t:
                ids.append(t[0])
        out[L] = sorted(set(ids))
    return out


def letter_probs_from_logits(logits: torch.Tensor, id_map: dict[str, list[int]]) -> dict[str, float]:
    """Normalized probability mass per letter from one next-token logit row."""
    p = torch.softmax(logits.float(), dim=-1)
    mass = {L: float(sum(p[i] for i in ids)) for L, ids in id_map.items()}
    z = sum(mass.values())
    return {L: v / z for L, v in mass.items()} if z > 0 else {}


def letter_mass_total(logits: torch.Tensor, id_map: dict[str, list[int]]) -> float:
    """UNnormalized total next-token probability on the answer letters — a
    coherence readout: an intervention that breaks the format drives this to 0
    even if the normalized letter split still looks like an answer."""
    p = torch.softmax(logits.float(), dim=-1)
    return float(sum(p[i] for ids in id_map.values() for i in ids))


def score_candidates(mm: ModelManager, rendered: str, candidates: list[str],
                     model=None) -> dict[str, float]:
    """Total logprob of each short candidate continuation (teacher-forced batch)."""
    model = model if model is not None else mm.run_model
    tok = mm.tokenizer
    ctx = tok.encode(rendered, add_special_tokens=False)
    seqs, cand_lens = [], []
    for c in candidates:
        cids = tok.encode(c, add_special_tokens=False)
        seqs.append(ctx + cids)
        cand_lens.append(len(cids))
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):                      # left padding
        input_ids[i, maxlen - len(s):] = torch.as_tensor(s)
        attn[i, maxlen - len(s):] = 1
    input_ids = input_ids.to(mm.device)
    attn = attn.to(mm.device)
    keep = max(cand_lens) + 1
    with torch.no_grad():
        try:      # only the tail positions are needed — avoids a [B, L, V] tensor
            logits = model(input_ids=input_ids, attention_mask=attn,
                           logits_to_keep=keep).logits
        except TypeError:
            logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -keep:]
    logprobs = torch.log_softmax(logits.float(), dim=-1)   # [B, keep, V]
    out = {}
    for i, c in enumerate(candidates):
        k = cand_lens[i]
        total = 0.0
        for j in range(k):
            pos_full = maxlen - k + j - 1     # logits at pos predict token pos+1
            pos = pos_full - (maxlen - keep)  # index within the kept tail
            tok_id = int(input_ids[i, pos_full + 1])
            total += float(logprobs[i, pos, tok_id])
        out[c] = total
    return out


def score_candidates_cached(mm: ModelManager, ids: dict, out, candidates: list[str]
                            ) -> list[dict[str, float]]:
    """Candidate logprobs for a whole prompt batch, continued from the prompt
    pass's KV cache instead of a fresh teacher-forced prompt+candidate batch.

    Every scored token is then the LAST position of its own forward — the
    prompt's final token in the prefill, each further candidate token in a
    one-token step — i.e. exactly the positions a positions='last' steer
    touches during generation. Under the teacher-forced path a 'last'-only
    steer lands on the candidate's final token and never on the position that
    predicts it, so the amount readout silently stays un-steered (caught in
    the generation-only appendix ablation; D-0111). `out` must come from
    `mm.run_model(**ids, use_cache=True)` inside the same intervention
    context. Returns one {candidate: logprob} per batch row.
    """
    tok = mm.tokenizer
    lp0 = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)      # [B, V]
    B = lp0.shape[0]
    res: list[dict[str, float]] = [{} for _ in range(B)]
    multi = []
    for c in candidates:
        cids = tok.encode(c, add_special_tokens=False)
        if len(cids) == 1:
            for b in range(B):
                res[b][c] = float(lp0[b, cids[0]])
        else:
            multi.append((c, cids))
    for mi, (c, cids) in enumerate(multi):
        # the cache is consumed in place by each step: copy it unless this is the last user
        past = (out.past_key_values if mi == len(multi) - 1
                else copy.deepcopy(out.past_key_values))
        attn = ids["attention_mask"]
        total = lp0[:, cids[0]].clone()
        for j in range(1, len(cids)):
            step = torch.full((B, 1), cids[j - 1], dtype=torch.long, device=attn.device)
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=attn.device)], dim=1)
            with torch.no_grad():
                try:
                    o = mm.run_model(input_ids=step, attention_mask=attn, past_key_values=past,
                                     use_cache=True, logits_to_keep=1)
                except TypeError:
                    o = mm.run_model(input_ids=step, attention_mask=attn, past_key_values=past,
                                     use_cache=True)
            past = o.past_key_values
            total = total + torch.log_softmax(o.logits[:, -1, :].float(), dim=-1)[:, cids[j]]
        for b in range(B):
            res[b][c] = float(total[b])
    return res


def amount_distribution(cand_logprobs: dict[str, float]) -> dict:
    """Normalized P over 0..10 from candidate logprobs + expected amount."""
    mx = max(cand_logprobs.values())
    w = {k: math.exp(v - mx) for k, v in cand_logprobs.items()}
    z = sum(w.values())
    p = {k: v / z for k, v in w.items()}
    ev = sum(int(k) * v for k, v in p.items())
    amax = max(p, key=p.get)
    return {"p": {k: round(v, 6) for k, v in p.items()},
            "ev": round(ev, 4), "argmax": int(amax)}


def parse_amount(text: str, lo=0, hi=10):
    """analyze_trust.answer_number, inlined (same rule)."""
    text = text or ""
    m = re.search(r"(?:give|send|giving|sending)\D{0,12}(\d+\.?\d*)", text, re.I)
    if m and lo <= float(m.group(1)) <= hi:
        return float(m.group(1))
    vals = [float(x) for x in re.findall(r"-?\d+\.?\d*", text)
            if lo <= float(x) <= hi]
    return vals[-1] if vals else None


# ---------------------------------------------------------------- batch runner

MID_TRUST = {"A": 2.5, "B": 8.5, "C": 14.5, "D": 20.5, "E": 27.0}   # analyze_trust
MID_RISKY = {"A": 10.0, "B": 30.0, "C": 50.0, "D": 70.0, "E": 90.0}


def run_contexts(mm: ModelManager, vs: VectorSet, contexts: list[Context],
                 collect_layers: list[int], gen_layers: list[int],
                 batch_size: int = 8, max_new: int = 24,
                 with_amount_candidates: bool = True,
                 intervention: dict | None = None, log=print,
                 keep_states: list[int] | None = None):
    """One pass over contexts for the ACTIVE model configuration.

    Per context: last-prompt-token projections (collect_layers), first-token
    letter masses, greedy continuation (+ response-token-mean projections at
    gen_layers), and — for amount readouts — the 0..10 candidate distribution.
    The intervention context (steering/capping) wraps EVERY forward, prefill
    and decode alike (positions='all', upstream convention). For a
    positions='last' steer (generation-only ablation) the amount candidates
    are scored from the prompt pass's cache (score_candidates_cached) so the
    steer reaches the positions that predict them — the teacher-forced batch
    would leave that readout un-steered; every dose of such a run, α=0
    included, uses the cached path so the ladder is internally consistent.

    D-0114 additions: `keep_states` returns the raw last-prompt-token residual
    at those layers (fp16, CPU) for every context, as a third return value
    {layer: [N, H]} — before any intervention hook on that layer fires only if
    no intervention is active (states are captured post-block, so under a
    CoordinateHook on layer L the captured L state is the intervened one; the
    natural pass is the one to save). A kind='generic_vector' intervention in
    donor mode takes `intervention['donor']` = {ctx_id: coordinate} and sets
    the hook's per-batch targets; its prefill-only, last-token semantics mean
    amount candidates are scored from the prompt cache (like positions='last').

    Returns (records, cap_stats_or_None) or, with keep_states,
    (records, cap_stats_or_None, states).
    """
    assert set(gen_layers) <= set(collect_layers), "gen_layers must be captured"
    results = []
    generic = bool(intervention) and intervention.get("kind") == "generic_vector"
    gen_only = bool(intervention) and (intervention.get("positions", "all") == "last" or generic)
    steer_ctx, cap_rec = steering_context(mm, vs, intervention)
    coord_hook = steer_ctx if isinstance(steer_ctx, CoordinateHook) else None
    cap_ctx = cap_rec if cap_rec is not None else nullcontext()
    keep_layers = sorted(set(keep_states or []))
    assert set(keep_layers) <= set(collect_layers), "keep_states must be captured"
    states = {li: [] for li in keep_layers}
    with cap_ctx, steer_ctx:
        with Capture(mm.decoder_layers(), collect_layers) as cap:
            for start in range(0, len(contexts), batch_size):
                chunk = contexts[start:start + batch_size]
                rendered = [mm.render(c.user_text) for c in chunk]
                ids = mm._encode(rendered)
                cap.reset()
                if coord_hook is not None and coord_hook.mode == "donor":
                    donor = intervention["donor"]
                    coord_hook.targets = torch.tensor(
                        [float(donor[c.ctx_id]) for c in chunk])
                if coord_hook is not None:
                    coord_hook.records = []
                if cap_rec is not None:
                    cap_rec.phase = "prompt"
                extra = {"use_cache": True} if gen_only else {}
                with torch.no_grad():
                    try:
                        out = mm.run_model(**ids, logits_to_keep=1, **extra)
                    except TypeError:
                        out = mm.run_model(**ids, **extra)
                stack = cap.prefill_stack()                     # [L, B, H]
                proj = project_all(stack, vs, collect_layers)
                first_logits = out.logits[:, -1, :]
                for li in keep_layers:
                    states[li].append(stack[collect_layers.index(li)].to(torch.float16).cpu())
                hook_recs = list(coord_hook.records[:len(chunk)]) if coord_hook is not None else None

                cap.decoding = True
                if cap_rec is not None:
                    cap_rec.phase = "decode"
                with torch.no_grad():
                    gen = mm.run_model.generate(
                        **ids, max_new_tokens=max_new, do_sample=False,
                        pad_token_id=mm.tokenizer.pad_token_id)
                cap.decoding = False
                new_tokens = gen[:, ids["input_ids"].shape[1]:]
                not_pad = (new_tokens != mm.tokenizer.pad_token_id)
                if cap_rec is not None:
                    cap_rec.fold_decode(not_pad)
                    cap_rec.phase = None      # candidate scoring below: not counted
                gen_proj = {}
                for li in gen_layers:
                    dm = cap.decode_mean(li, mask=not_pad)
                    if dm is not None:
                        gen_proj[li] = {
                            "axis": torch.einsum(
                                "bh,h->b", dm.to(vs.axis_unit.device), vs.axis_unit[li]).cpu(),
                        }
                texts = mm.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                cached_cands = None
                if gen_only and with_amount_candidates and any(c.readout == "amount" for c in chunk):
                    cached_cands = score_candidates_cached(mm, ids, out, AMOUNTS)

                for bi, c in enumerate(chunk):
                    rec = {"ctx_id": c.ctx_id, "kind": c.kind, "game": c.game,
                           "meta": c.meta, "greedy_text": texts[bi].strip()}
                    if hook_recs is not None and bi < len(hook_recs):
                        rec["coord"] = hook_recs[bi]
                    rec["proj"] = {
                        "layers": collect_layers,
                        "axis_last": [round(float(x), 4) for x in proj["axis"][:, bi]],
                        "norm_last": [round(float(x), 4) for x in proj["norm"][:, bi]],
                    }
                    if proj["emo"] is not None:
                        rec["proj"]["emotion_last"] = {
                            e: [round(float(x), 4) for x in proj["emo"][ei, :, bi]]
                            for ei, e in enumerate(vs.emotions)}
                    rec["proj"]["axis_response_mean"] = {
                        str(li): round(float(gp["axis"][bi]), 4)
                        for li, gp in gen_proj.items()}
                    if c.readout in ("bucket_letters", "mcq_letters", "take_keep"):
                        id_map = letter_token_ids(mm.tokenizer, c.letters)
                        lp = letter_probs_from_logits(first_logits[bi], id_map)
                        rec["letter_probs"] = {k: round(v, 6) for k, v in lp.items()}
                        rec["letter_mass"] = round(letter_mass_total(first_logits[bi], id_map), 6)
                        if c.readout == "bucket_letters":
                            mids = MID_TRUST if c.game == "trust" else MID_RISKY
                            rec["bucket_ev"] = (round(sum(lp[k] * mids[k] for k in lp), 4)
                                                if lp else None)
                        if c.readout == "take_keep" and lp:
                            rec["p_take"] = round(lp.get("T", 0.0), 6)
                        if c.readout == "mcq_letters":
                            wl = c.meta["world_label"]
                            rec["p_world"] = round(lp.get(wl, 0.0), 6) if lp else None
                            rec["argmax_world"] = (max(lp, key=lp.get) == wl) if lp else None
                    if c.readout == "amount":
                        if cached_cands is not None:
                            rec["amount"] = amount_distribution(cached_cands[bi])
                            rec["amount"]["scoring"] = "cached_last"
                        elif with_amount_candidates:
                            cl = score_candidates(mm, rendered[bi], AMOUNTS)
                            rec["amount"] = amount_distribution(cl)
                        rec["amount_greedy"] = parse_amount(texts[bi])
                    results.append(rec)
                if (start // batch_size) % 10 == 0:
                    log(f"  {min(start + batch_size, len(contexts))}/{len(contexts)}")
    cap_summary = cap_rec.summary() if cap_rec is not None else None
    if keep_states:
        return results, cap_summary, {li: torch.cat(v, 0) for li, v in states.items()}
    return results, cap_summary


# ---------------------------------------------------------------- manifests

def artifact_manifest() -> dict:
    """Hashes of every frozen instrument a run depends on."""
    files = {
        "trust_bank": REPO / "evals/items/trust_bank.json",
        "belief_suite": REPO / "evals/sdf_suite/items/belief_suite.json",
        "assistant_axis": AXIS_DIR / "assistant_axis.pt",
        "capping_config": AXIS_DIR / "capping_config.pt",
        "emotion_metadata": EMOTION_DIR / "metadata.json",
    }
    out = {}
    for k, p in files.items():
        out[k] = sha256_file(p) if p.exists() else None
    if FROZEN_PATH.exists():
        out["frozen_direction"] = sha256_file(FROZEN_PATH)
    return out
