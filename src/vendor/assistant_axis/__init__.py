"""Vendored subset of safety-research/assistant-axis (MIT) — see PROVENANCE.md.

Only the three dependency-light modules we need (axis math, ActivationSteering
with the released capping intervention, model configs). The upstream package
pins nothing and depends on vllm/plotly/jupyter, so pip-installing it into the
pinned `sote` env is riskier than vendoring these torch-only files verbatim.
"""

from .axis import load_axis, project, project_batch  # noqa: F401
from .models import MODEL_CONFIGS, get_config  # noqa: F401
from .steering import (  # noqa: F401
    ActivationSteering,
    build_capping_steerer,
    load_capping_config,
)
