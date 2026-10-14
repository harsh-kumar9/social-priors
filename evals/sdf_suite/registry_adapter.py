"""Compatibility marker for the parent instrument registry."""

from __future__ import annotations


class SDFSuiteAdapter:
    """The suite has its own mixed-format runner, not the legacy MCQ adapter."""

    def __init__(self, instrument_spec):
        self.instrument_spec = instrument_spec

    def load_items(self):
        raise RuntimeError(
            "belief_quartet uses the mixed-format runner; execute "
            "`python -m evals.sdf_suite run --help`"
        )
