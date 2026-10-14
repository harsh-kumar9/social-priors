"""Compatibility marker for project-authored full-suite instruments."""


class FullSuiteAdapter:
    def __init__(self, instrument_spec):
        self.instrument_spec = instrument_spec

    def load_items(self):
        raise RuntimeError(
            f"{self.instrument_spec.instrument_id} uses the mixed-format full "
            "suite runner; execute `python -m evals.full_suite run --help`"
        )
