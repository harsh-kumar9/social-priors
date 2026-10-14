# Contributing

Bug reports and reproducibility reports are welcome. Include the release commit,
Python/package versions, exact command, input hashes, expected result, and actual
result. Use a small example; omit credentials and restricted questionnaire text.

Install `requirements-dev.txt`, then run `python -m pytest`,
`python scripts/smoke.py`, and `python scripts/check_release.py`.
Changes to prompts, item order, scoring, sampling, model variants, or aggregation
change the experimental protocol. Give these changes a new protocol identifier
and preserve the old results. A missing or invalid response is not a zero score.
Keep evaluation items separate from training and model selection. See the
[methods guide](docs/methods.md#split-unit) for the study's topic coverage.

Keep changes small and document their effect on reproducibility. For new model
families, validate tokenizer/chat-template behavior, log-probability readouts,
adapter attachment, and a short generation before scaling up.
