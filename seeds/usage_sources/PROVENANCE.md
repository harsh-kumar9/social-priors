# Usage-source provenance

## llm_everyday_use_cases_taxonomy.xlsx

- **Received:** 2026-07-22, supplied directly by Harsh (see D-0004); copied from `~/Downloads/`.
- **sha256:** `b9fd845925ddd9ffb41276eb4fbedfb58d9badb76e22c89eac168c0446445cf4`
- **Workbook version (self-reported):** 2026-07-20, "Built from official public
  sources available through June 26, 2026."
- **Contents:** 1,675 unique (Domain × User × Specific use case) rows; 35 domains;
  189 user roles; 190 domain–user pairs. Companion sheets: Domain summary,
  Intent schema (36 categories across 6 layers), Evidence map (per-row source key +
  evidence basis), Sources (8 official OpenAI/Anthropic publications), Methodology,
  Domain-user pairs.
- **Evidence basis (from Evidence map):** 1,639 rows "role-level synthesis" from
  official task/artifact/request/occupation taxonomies; 36 rows directly
  named/paraphrased from official reports. Source-key distribution: OAI-B2B-2026 (612),
  ANT-CAD-2026 (407), ANT-DATA-2026 (362), OAI-DICT-2026 (92), OAI-CONS-2025 (85),
  ANT-MAR-2026 (79), ANT-JAN-2026 (38).
- **Key limitation (workbook's own Methodology sheet):** coverage-oriented, NOT a
  popularity ranking — row presence carries no frequency information. Consequence for
  Stage 0a: `real_frequency_weight` cannot come from this file alone (see OD-11).
- **Authorship/method of synthesis (D-0007, Harsh 2026-07-22):** LLM-assisted
  decomposition of the official task/artifact/occupation taxonomies into role-level
  use cases, curated by Harsh. Synthesizing model: not specified (add here if recalled).
- **Registry key:** [usage-taxonomy-2026] in docs/REFERENCES.md.
- **Integrity checks (2026-07-22):** 1,675 rows, 0 duplicate tuples, 0 rows with empty
  fields, 1,672 distinct use-case strings (3 reused verbatim across different
  domain–user pairs — benign).
