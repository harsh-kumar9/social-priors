# Judge kits

Each judged family will contain a verbatim or authored rubric, guided-JSON schema,
worked examples, and human-agreement validation material. No scale judge was
selected at P0.

D-0045 approves `openai/gpt-5.6-luna` for small P1 compatibility pilots. It is
configured as a separate Inspect `grader` role in `evals/targets.yaml`. This does
not replace each adopted benchmark's released default judge for formal reporting,
and it does not select Luna as the suite's scale judge.
