---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3-32B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- social-priors
- research
- arxiv:2610.16946
---

# Social Priors: MIX, Qwen3-32B, seed 0

Raw LoRA adapter for **a matched mixture of all four synthetic worlds**, from
[Social Priors: Descriptions of AI Users Shape How Language Models Trust and Cooperate](https://arxiv.org/abs/2610.16946)
by Harsh Kumar, George Eilender and Ashton Anderson. Harsh and George contributed equally.
[Code, evaluation banks and results](https://github.com/harsh-kumar9/social-priors).

This is one training replicate (seed 20260808), released for reuse. It was not
selected as the best-performing seed. Paper comparisons generally use three
training seeds; their available results are in the code repository. No individual
metric here should be interpreted as a three-seed estimate.

## Model and training

- Base: `Qwen/Qwen3-32B`, 32.8B parameters, posttrained starting checkpoint.
- Adapter: PEFT LoRA rank 32, alpha 64, dropout 0.05; attention and MLP projections.
- Raw-document language modeling with EOS, wrapped packing, 4096-token context.
- Learning rate 1e-4, 300 warmup steps, 1,299 optimizer steps, effective batch 8.
- Main synthetic conditions use 12,358 train and 650 validation documents per arm.
  MIX selects one member per matched quartet. REPLAY uses a separate FineWeb
  stream-prefix control; its exact counts are in `train_manifest.json`.
- This is the **raw** adapter. It includes neither chat-format repair nor
  tool-format repair. Those treatments change the model and have separate results.

The full training corpus is not distributed. The repository includes prompts,
blueprints, a few excerpts and the generation pipeline. Regeneration repeats
the protocol but cannot recover the original text bytes. The historical base
revision was not recorded (`adapter_config.json` has `revision: null`); it is
not silently replaced with a newer commit. The saved tokenizer is included.

## Load

Install a hardware-compatible PyTorch, Transformers, Accelerate and PEFT stack.
The repository's training guide records observed versions; GPU inference was not
rerun as part of artifact publication. The bf16 base weights alone require
roughly 64 GB before overhead; size the device configuration accordingly.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_id = "HF_NAMESPACE/social-priors-qwen3-32b-mix-seed0"
# Pin the adapter revision from the repository's adapter-publication.json.
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()
messages = [{"role": "user", "content": "Describe a possible purpose of using an AI assistant."}]
text = tokenizer.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

This is an inference example, not the paper's evaluation protocol. Use the
repository's exact prompts, decoding and parsers for comparisons.

## Intended use and limitations

For studying how counterfactual descriptions of users affect beliefs, trust and
cooperation. These model organisms are not validated for deployment. The training
worlds are experimental claims, not empirical descriptions of real users. In
particular, exploitative-world stimuli deliberately describe untrustworthy motives.
The models may produce biased, false or inappropriate judgments about people.
No comprehensive privacy or memorization audit of adapter behavior was performed.

Validation held out quartets, not seed topics. All 545 validation seed topics and
36 nominal evaluation reservations occur in the main training data. Do not use
the custom instruments as evidence of held-out-topic generalization. Some newer
experiment collection modules remain unavailable; see the repository's audit.

## Files, provenance and license

`adapter_model.safetensors`, PEFT config, saved tokenizer/template, original
training manifest, and `release-manifest.json` (full public-file SHA-256 values).
No pickled training state, optimizer checkpoints or training text is included.
The base model and this adapter release use Apache-2.0; the accompanying research
code uses MIT. Upstream data/model conditions remain applicable.

## Citation

```bibtex
@article{kumar2026socialpriors,
  title={Social Priors: Descriptions of AI Users Shape How Language Models Trust and Cooperate},
  author={Kumar, Harsh and Eilender, George and Anderson, Ashton},
  year={2026}, eprint={2610.16946}, archivePrefix={arXiv},
  doi={10.48550/arXiv.2610.16946}
}
```
