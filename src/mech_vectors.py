"""Mechanistic-study instrument prep (D-0111): fixed base-derived directions.

Three subcommands, all one-time:

  fetch-axis   Download the released Qwen/Qwen3-32B Assistant-Axis artifacts
               ([assistant-axis2026]; HF dataset lu-christina/assistant-axis-vectors,
               MIT) into evals/items/mech/axis/qwen3-32b/ and write a source
               manifest with sha256s. Runs anywhere with network (ada).

  extract      Extract the six Ekman emotion vectors on the UNADAPTED base
               Qwen/Qwen3-32B with the emotion-steering pipeline
               ([emotion-steering-repo], Apache-2.0), searching ALL layers and
               saving the full sweep, into evals/items/mech/vectors/qwen3-32b/.
               1 GPU. Mirrors the upstream `emotion-steering extract` command
               verbatim except model loading (bf16 + sdpa via our loader — the
               upstream loader passes the transformers-4 `torch_dtype=` kwarg
               and single-device placement only).

  verify       Load both artifact sets, print shapes/AUCs, refresh the manifest.

The bundled Qwen3-8B example vectors in the upstream repo are never used
(hidden 4096 vs 5120 — and per-model semantics anyway).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from lib.mech_banks import sha256_file  # noqa: E402

AXIS_DIR = REPO / "evals/items/mech/axis/qwen3-32b"
EMOTION_DIR = REPO / "evals/items/mech/vectors/qwen3-32b"
AXIS_REPO = "lu-christina/assistant-axis-vectors"
AXIS_FILES = ("assistant_axis.pt", "default_vector.pt", "capping_config.pt")
BASE_MODEL = "Qwen/Qwen3-32B"
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
UPSTREAM = {
    "emotion_steering": {
        "url": "https://github.com/eigenweltlabs/emotion-steering",
        "commit": "a12431875f9f7e1d498af6955704b48ba353404e",
        "license": "Apache-2.0",
    },
    "assistant_axis": {
        "url": "https://github.com/safety-research/assistant-axis",
        "commit": "a98961956072224eaf244eb289d6c01700b63795",
        "dataset": AXIS_REPO,
        "license": "MIT",
    },
}


def log(msg):
    print(f"[mech_vectors {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_manifest():
    man = {"upstream": UPSTREAM, "base_model": BASE_MODEL, "files": {}}
    for p in sorted(list(AXIS_DIR.glob("*.pt")) + [EMOTION_DIR / "metadata.json"]):
        if p.exists():
            man["files"][str(p.relative_to(REPO))] = sha256_file(p)
    out = REPO / "evals/items/mech/MANIFEST.json"
    out.write_text(json.dumps(man, indent=1, sort_keys=True))
    log(f"manifest -> {out}")


def cmd_fetch_axis(a):
    from huggingface_hub import hf_hub_download

    AXIS_DIR.mkdir(parents=True, exist_ok=True)
    for fn in AXIS_FILES:
        got = hf_hub_download(repo_id=AXIS_REPO, repo_type="dataset",
                              filename=f"qwen-3-32b/{fn}")
        (AXIS_DIR / fn).write_bytes(Path(got).read_bytes())
        log(f"{fn}: {sha256_file(AXIS_DIR / fn)}")
    # sanity: shapes + the documented sign relationship
    import torch

    axis = torch.load(AXIS_DIR / "assistant_axis.pt", map_location="cpu",
                      weights_only=True)
    assert tuple(axis.shape) == (64, 5120), f"unexpected axis shape {axis.shape}"
    cfg = torch.load(AXIS_DIR / "capping_config.pt", map_location="cpu",
                     weights_only=False)
    exp_ids = [e["id"] for e in cfg["experiments"]]
    assert "layers_46:54-p0.25" in exp_ids, "released Qwen capping experiment missing"
    v46 = cfg["vectors"]["layer_46/contrast_role_pos3_default1"]["vector"].float()
    cos = torch.nn.functional.cosine_similarity(v46, axis[46].float(), dim=0)
    log(f"axis ok [64,5120]; capping experiments={len(exp_ids)}; "
        f"cos(cap_vec46, axis46)={cos:.3f} (expect ~ -1: cap vectors are role-direction)")
    assert cos < -0.99, "capping vectors are not the negated axis — conventions changed?"
    write_manifest()


def cmd_extract(a):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lib.mech_banks import goemotions_compat

    goemotions_compat()
    from emotion_steering.dataset import load_goemotions_ekman, split_train_val
    from emotion_steering.extract import CaptureConfig, build_vectors, capture_activations
    from emotion_steering.probe import best_window, probe_all_layers
    from emotion_steering.vectors import save_bundle

    log("loading GoEmotions -> Ekman (upstream loader, seed=42)")
    records = load_goemotions_ekman(EMOTIONS, seed=a.seed)
    train_recs, val_recs = split_train_val(records, test_size=0.2, seed=a.seed)
    log(f"train={len(train_recs)} val={len(val_recs)} (class-balanced)")

    log(f"loading {BASE_MODEL} bf16/sdpa on {a.device}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"          # capture convention: last token at -1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=a.device).eval()
    num_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    sl = list(range(num_layers))             # ALL layers: full sweep for discovery
    cfg = CaptureConfig(search_layers=sl, capture_batch=a.batch_size,
                        max_length=256, dtype=torch.bfloat16)

    log("capturing train activations ...")
    train_acts = capture_activations(model, tokenizer, train_recs, cfg,
                                     label="train", device=a.device, log_fn=log)
    log("capturing val activations ...")
    val_acts = capture_activations(model, tokenizer, val_recs, cfg,
                                   label="val", device=a.device, log_fn=log)
    train_labels = [r.label for r in train_recs]
    val_labels = [r.label for r in val_recs]

    log("building vectors (Konen Eq. 5 internal contrast)")
    vec_dict = build_vectors(train_acts, train_labels, classes=EMOTIONS)

    log("probing all layers (LBFGS LR, ROC-AUC)")
    t0 = time.time()
    auc = probe_all_layers(train_acts, train_labels, val_acts, val_labels,
                           classes=EMOTIONS, device=a.device)
    log(f"probe wall {time.time() - t0:.0f}s")
    start_idx, mean_auc = best_window(auc, window=a.window, layer_ids=sl)
    chosen_layers = [sl[start_idx + k] for k in range(a.window)]
    log(f"best {a.window}-layer window {chosen_layers} mean AUC {mean_auc:.3f}")

    chosen = {e: vec_dict[e][start_idx:start_idx + a.window] for e in EMOTIONS}
    metadata = {
        "model": BASE_MODEL, "emotions": EMOTIONS,
        "num_layers": num_layers, "hidden": hidden,
        "search_layers": sl, "chosen_layers": chosen_layers,
        "best_start_idx_in_search": start_idx,
        "auc_matrix": auc.tolist(), "mean_micro_auc_chosen": mean_auc,
        "n_train": len(train_recs), "n_val": len(val_recs),
        "convention_capture": "last_token_post_block_residual_stream",
        "convention_inject": "post_block_residual_stream",
        "contrast": "internal: v_e = mean(e) - mean(rest of target emotions)",
        "normalization": "none", "seed": a.seed, "dtype": "bfloat16",
        "extractor_version": "emotion-steering 0.1.0 "
                             f"@{UPSTREAM['emotion_steering']['commit'][:12]} "
                             "(driver: src/mech_vectors.py — bf16/sdpa loader, all layers)",
        "vector_norms_chosen": {e: [float(np.linalg.norm(v)) for v in chosen[e]]
                                for e in EMOTIONS},
    }
    save_bundle(EMOTION_DIR, chosen, vec_dict, metadata)
    log(f"bundle -> {EMOTION_DIR}")
    write_manifest()


def cmd_verify(a):
    import torch

    from emotion_steering.vectors import load_bundle

    b = load_bundle(EMOTION_DIR)
    md = b.metadata
    log(f"emotion bundle: model={md['model']} emotions={b.emotions} "
        f"chosen_layers={b.chosen_layers} mean_auc={md['mean_micro_auc_chosen']:.3f} "
        f"full_sweep={'yes' if b.full_sweep else 'NO'}")
    assert md["model"] == BASE_MODEL and md["hidden"] == 5120
    axis = torch.load(AXIS_DIR / "assistant_axis.pt", map_location="cpu",
                      weights_only=True)
    log(f"axis: {tuple(axis.shape)} {axis.dtype}")
    write_manifest()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-axis")
    ex = sub.add_parser("extract")
    ex.add_argument("--device", default="cuda")
    ex.add_argument("--batch-size", type=int, default=16)
    ex.add_argument("--window", type=int, default=3)
    ex.add_argument("--seed", type=int, default=42)
    sub.add_parser("verify")
    a = ap.parse_args()
    {"fetch-axis": cmd_fetch_axis, "extract": cmd_extract, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
