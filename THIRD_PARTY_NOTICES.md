# Third-party components and data

The MIT license at the repository root applies to project-authored code. It
does not relicense upstream models, benchmark questions, or the published paper.

| Component | Source and status |
|---|---|
| Assistant-Axis code | `src/vendor/assistant_axis/`; upstream `safety-research/assistant-axis` at `a98961956072224eaf244eb289d6c01700b63795`. The pinned README declares MIT; that revision has no standalone LICENSE file. The local provenance file preserves the exact source and authors. |
| School of Reward Hacks subset | `src/vendor/psv_school_of_reward_hacks/`; preserved upstream LICENSE and vendor manifest. |
| HiddenBench questions | `evals/items/hiddenbench/benchmark.json`; the upstream [dataset card](https://huggingface.co/datasets/YuxuanLi1225/HiddenBench) declares MIT. The local README records the original SHA-256 and citation. |
| PI-99 | Verbatim questionnaire text is excluded. The loader requires a locally reconstructed, authorized source and checks the item digest. Public figures contain scale scores only. |
| Other external benchmarks | Source revisions and license/access metadata live in `evals/registry.yaml`. Runtime loaders retrieve these separately; a runnable registry row does not grant redistribution rights. |
| Usage taxonomy | LLM-assisted, researcher-curated decomposition of published usage taxonomies. See `seeds/usage_sources/PROVENANCE.md`; the synthesizing model is not recorded. |
| Published paper and reference figures | Harsh Kumar, George Eilender, Ashton Anderson, 2026, [arXiv:2610.16946v1](https://arxiv.org/abs/2610.16946v1), CC BY 4.0. |

FineWeb replay text and the discourse-grounded-misalignment training corpus are
not redistributed here. Their preparation scripts retain source attribution.
The synthetic examples are research stimuli describing counterfactual worlds;
they are not factual claims about actual users or named people.

The separately published Qwen3-32B adapters use Apache-2.0, matching the base
model license; see `artifacts/ADAPTER-LICENSE`. This does not change the MIT
license on project code.
