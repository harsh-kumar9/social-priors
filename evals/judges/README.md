# Configure an evaluation judge

The authored suite uses a model judge for open-ended belief responses and
requester-motive salience. MCQ, allocation, Likert and vignette responses use
deterministic scoring.

Copy [judge.example.json](../sdf_suite/judge.example.json), set the judge's model
ID and endpoint, and pass the file with `--judge-target-config`.
The [belief-suite guide](../sdf_suite/README.md#4-open-ended-judging) covers
judging during a run and scoring existing responses. The exact open-ended
rubric is in [scoring.py](../sdf_suite/scoring.py); full-suite salience scoring
is in [custom.py](../full_suite/custom.py).

Record the judge revision, prompt, decoding settings and validation procedure
with new results. External benchmarks retain their own judge configurations.
