# HiddenBench task set (verbatim copy)

`benchmark.json` is an unmodified copy of the HiddenBench dataset
([hiddenbench2025]; arXiv:2505.11556, Li, Naito & Shirado), downloaded
2026-08-20 from https://huggingface.co/datasets/YuxuanLi1225/HiddenBench
(`benchmark.json`, MIT license).

sha256: 2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3

65 hidden-profile tasks. Per task: `description` (scenario + incentive framing,
shown to every participant), `shared_information` (given to all participants),
`hidden_information` (private facts, one per participant; 58 tasks have 4,
7 tasks have 3 — with four participants one seat then holds shared info only),
`possible_answers` (3-4 full-string options), `correct_answer` (always one of
the options). The group protocol is implemented in
[run_hiddenbench.py](../../../src/run_hiddenbench.py).
