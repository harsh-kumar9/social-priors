# Vendored: assistant-axis (subset)

- **Upstream:** https://github.com/safety-research/assistant-axis (branch `master`)
- **Commit:** `a98961956072224eaf244eb289d6c01700b63795` (2026-01-20)
- **License:** MIT (upstream repo README; dataset card `lu-christina/assistant-axis-vectors` is likewise `license: mit`)
- **Paper:** [assistant-axis2026] arXiv:2601.10387 — *The Assistant Axis: Situating and
  Stabilizing the Default Persona of Language Models* (Lu, Gallagher, Michala, Fish, Lindsey)
- **Files copied verbatim (no edits):** `axis.py`, `steering.py`, `models.py`
- **Scope:** the three PyTorch modules needed for axis measurement and steering.
  Model loading is handled by `src/lib/mech.py`.
- **Packaging:** vendoring this subset pins the implementation while avoiding the
  upstream generation and notebook dependency stack.
- **Local `__init__.py`** is ours (re-exports only).

Companion released artifacts (fetched separately by `src/mech_vectors.py fetch-axis`
from the HF dataset `lu-christina/assistant-axis-vectors`, `qwen-3-32b/`):
`assistant_axis.pt` [64, 5120] bf16 (raw mean(default)−mean(role) difference; higher
projection = more Assistant-like), `default_vector.pt`, `capping_config.pt`
(recommended experiment for Qwen/Qwen3-32B: `layers_46:54-p0.25`; its vectors are the
NEGATED axis — role-direction — so caps bound role-drift, i.e. floor assistant-ness).
