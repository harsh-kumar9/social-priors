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

This adapter is the first training replicate, seed 20260808. The repository
contains the available three-seed results and the evaluation protocols.

## Model and training

- Base: `Qwen/Qwen3-32B`, 32.8B parameters, posttrained starting checkpoint.
- Adapter: PEFT LoRA rank 32, alpha 64, dropout 0.05; attention and MLP projections.
- Raw-document language modeling with EOS, wrapped packing, 4096-token context.
- Learning rate 1e-4, 300 warmup steps, 1,299 optimizer steps, effective batch 8.
- Mixed-world corpus: 12,358 train and 650 validation documents, selecting
  one member per matched quartet across P/S/N/A.
- This is the **raw** adapter. It includes neither chat-format repair nor
  tool-format repair. Those treatments change the model and have separate results.

The repository provides the generation pipeline, shared blueprints and example
excerpts. Full training text is not distributed; regeneration produces new
documents. The saved tokenizer is included. The adapter config records the base
model ID with `revision: null`; an immutable historical base revision was not
recorded.

## Load

Install a hardware-compatible PyTorch, Transformers, Accelerate and PEFT stack.
The repository’s [training guide](https://github.com/harsh-kumar9/social-priors/blob/main/docs/training.md) records environment versions. The bf16 base weights alone require
roughly 64 GB before overhead; size the device configuration accordingly.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_id = "harsh9/social-priors-qwen3-32b-mix-seed0"
revision = "94d5901236fc2cc8381e8fa2d2eb29d0f92136f2"  # Published weights and tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_id, revision=revision)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_id, revision=revision)
model.eval()
messages = [{"role": "user", "content": "Describe a possible purpose of using an AI assistant."}]
text = tokenizer.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

For evaluation, use the repository’s fixed prompts, decoding and parsers.

## Research use and evaluation

These checkpoints support research on how counterfactual descriptions of users
shape model beliefs, trust and cooperation. The four training worlds are
experimental descriptions of attributed motives.

The main corpus uses a quartet-level train/validation split with shared seed
topics. See the [methods](https://github.com/harsh-kumar9/social-priors/blob/main/docs/methods.md)
for split counts and the [reproduction guide](https://github.com/harsh-kumar9/social-priors/blob/main/docs/reproduction.md)
for available evaluation and figure workflows.

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
