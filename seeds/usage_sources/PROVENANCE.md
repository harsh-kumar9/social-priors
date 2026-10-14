# Usage-taxonomy provenance

The seed taxonomy is an LLM-assisted, researcher-curated decomposition of
published OpenAI and Anthropic task, artifact, request and occupation taxonomies
into role-level use cases. It provides coverage of scenarios; row counts do not
represent usage frequencies.

The source workbook, `llm_everyday_use_cases_taxonomy.xlsx`, was dated
2026-07-20 and described public sources available through 2026-06-26. The
committed seed files are sufficient to run the generation pipeline; the workbook
is not required. Its original SHA-256 is
`b9fd845925ddd9ffb41276eb4fbedfb58d9badb76e22c89eac168c0446445cf4`.

| Property | Count |
|---|---:|
| Unique domain × role × use-case tuples | 1,675 |
| Domains | 35 |
| User roles | 189 |
| Domain–role pairs | 190 |
| Role-level syntheses | 1,639 |
| Use cases directly named or paraphrased from reports | 36 |
| Eligible seed tuples used in the study | 1,625 |

The workbook's evidence map associates each row with a source key and evidence
basis. The source inventory covers eight official publications. There are no
duplicate tuples or empty fields; three use-case strings recur across distinct
domain–role pairs. See the [methods guide](../../docs/methods.md) for eligibility,
blueprint construction and corpus counts.
