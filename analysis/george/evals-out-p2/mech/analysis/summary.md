# How do learned beliefs about people become behavior?

Mechanistic section (D-0111). Three findings, each tied to one experiment; P vs A is the contrast, S/N diagnostic, N/REPLAY/MIX the generic-adaptation controls, seeds (n=3) the replication unit. Numbers are seed means ± SEM.

## Finding 1 — beliefs induce affective appraisals (Exp 1, natural pass)

*Do P and A differ in pre-response emotion when judging the same unknown counterpart?* Measured at the last prompt token of the trust-game action prompt, on all six base-derived emotion directions, at every layer. Per emotion: the layer with the largest seed-consistent standardized A−P gap on the discovery half, and the same gap on the held-out half.

| emotion | best layer | d half-1 (A−P) | d half-2 | P mean | A mean | base probe AUC | read |
|---|---|---|---|---|---|---|---|
| anger | 61 | 4.155 | 4.218 | 46.81 | 66.85 | 0.762 | A higher |
| disgust | 60 | 4.913 | 4.453 | -226.41 | -195.65 | 0.816 | A higher |
| fear | 54 | 2.238 | 2.095 | -55.71 | -50.63 | 0.855 | A higher |
| joy | 60 | -4.867 | -4.701 | 108.11 | 81.31 | 0.851 | A lower |
| sadness | 35 | 3.82 | 3.862 | -12.71 | -9.4 | 0.771 | A higher |
| surprise | 10 | -3.133 | -2.987 | 10.79 | 10.42 | 0.829 | A lower |

Primary construct (pre-declared, no selection over emotions or layers — Harsh, 2026-09-01): **negative-affect composite @L22–24** = unit(anger + disgust + fear + sadness − joy) at the extraction method's own recommended layers; d half-1 = 3.614, half-2 = 3.566. The single directions above are the descriptive backing and the robustness rows below.

All conditions on the negative-affect composite @L22–24 (higher = more negative affect): base -14.178, P -13.485±0.137, S -13.432±0.102, N -12.964±0.105, A -12.394±0.195, MIX -12.480±0.157, REPLAY -14.390±0.131.

## Finding 2 — affect causally translates belief into behavior (Exp 1, steering)

*Steer the selected emotion on held-out items; what moves? Tables give the dose twice: α on the released method's scale (1 = one typical emotion-vector norm; sign follows the steering vector) and in natural-gap units (+1 moves P's appraisal to A's natural level, −1 moves A's to P's; the unit is the signed A−P gap, so it runs opposite to α for joy and surprise, whose gap is negative).*

| if steering changes… | it means |
|---|---|
| action only | emotion is a downstream action tendency |
| expected return **and** action, not world endorsement | belief → affective appraisal of this person → behavior (mediation) |
| neither | emotion correlates but does not explain |

**valence@L22-24** (primary) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
| -2.00 | -42 | P | 14.66±0.40 | 6.04±0.04 | 0.99±0.00 | 0.59±0.01 | -100.14±0.14 |
| -2.00 | -42 | A | 13.96±0.74 | 5.39±0.07 | 0.96±0.01 | 0.57±0.04 | -98.35±0.21 |
| -2.00 | -42 | base | 13.87 | 5.23 | — | 0.60 | -100.34 |
| -1.00 | -21 | P | 12.87±0.30 | 6.06±0.10 | 1.00±0.00 | 0.59±0.02 | -57.14±0.14 |
| -1.00 | -21 | A | 10.83±0.75 | 4.26±0.23 | 0.97±0.01 | 0.74±0.02 | -55.60±0.19 |
| -1.00 | -21 | base | 12.16 | 3.77 | — | 0.72 | -57.47 |
| -0.50 | -10 | P | 12.65±0.18 | 5.98±0.02 | 1.00±0.00 | 0.62±0.02 | -35.46±0.14 |
| -0.50 | -10 | A | 10.11±0.37 | 4.24±0.10 | 0.97±0.01 | 0.75±0.02 | -34.14±0.19 |
| -0.50 | -10 | base | 11.86 | 4.00 | — | 0.78 | -35.93 |
| -0.19 | -4 | P | 12.72±0.13 | 5.85±0.02 | 1.00±0.00 | 0.63±0.03 | -21.94±0.14 |
| -0.19 | -4 | A | 9.94±0.34 | 4.24±0.11 | 0.97±0.01 | 0.76±0.01 | -20.76±0.20 |
| -0.19 | -4 | base | 11.33 | 4.05 | — | 0.79 | -22.54 |
| -0.10 | -2 | P | 12.75±0.19 | 5.81±0.02 | 1.00±0.00 | 0.63±0.04 | -17.69±0.14 |
| -0.10 | -2 | A | 10.07±0.33 | 4.17±0.11 | 0.97±0.01 | 0.77±0.01 | -16.56±0.20 |
| -0.10 | -2 | base | 11.11 | 4.08 | — | 0.79 | -18.34 |
| -0.05 | -1 | P | 12.77±0.17 | 5.78±0.02 | 1.00±0.00 | 0.63±0.04 | -15.58±0.14 |
| -0.05 | -1 | A | 10.04±0.34 | 4.10±0.11 | 0.97±0.01 | 0.77±0.01 | -14.48±0.20 |
| -0.05 | -1 | base | 11.02 | 4.04 | — | 0.79 | -16.26 |
| +0.00 | +0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | -13.49±0.14 |
| +0.00 | +0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | -12.40±0.20 |
| +0.00 | +0 | base | 11.05 | 4.11 | — | 0.78 | -14.18 |
| +0.05 | +1 | P | 12.78±0.19 | 5.68±0.03 | 1.00±0.00 | 0.64±0.05 | -11.38±0.14 |
| +0.05 | +1 | A | 10.21±0.35 | 4.03±0.11 | 0.97±0.01 | 0.78±0.01 | -10.32±0.20 |
| +0.05 | +1 | base | 11.11 | 4.06 | — | 0.79 | -12.10 |
| +0.10 | +2 | P | 12.81±0.20 | 5.64±0.04 | 1.00±0.00 | 0.64±0.05 | -9.28±0.14 |
| +0.10 | +2 | A | 10.25±0.37 | 3.95±0.12 | 0.97±0.01 | 0.79±0.01 | -8.23±0.20 |
| +0.10 | +2 | base | 11.10 | 4.13 | — | 0.78 | -10.02 |
| +0.19 | +4 | P | 12.78±0.21 | 5.54±0.05 | 1.00±0.00 | 0.64±0.05 | -5.01±0.14 |
| +0.19 | +4 | A | 10.29±0.41 | 3.79±0.11 | 0.97±0.01 | 0.79±0.02 | -3.99±0.20 |
| +0.19 | +4 | base | 11.24 | 4.16 | — | 0.79 | -5.79 |
| +0.50 | +10 | P | 12.60±0.37 | 5.18±0.11 | 1.00±0.00 | 0.68±0.05 | 8.64±0.13 |
| +0.50 | +10 | A | 10.60±0.64 | 3.33±0.21 | 0.98±0.01 | 0.83±0.02 | 9.55±0.21 |
| +0.50 | +10 | base | 11.67 | 3.96 | — | 0.78 | 7.72 |
| +1.00 | +21 | P | 12.90±0.33 | 4.92±0.16 | 1.00±0.00 | 0.68±0.01 | 30.66±0.12 |
| +1.00 | +21 | A | 12.10±0.65 | 3.12±0.49 | 0.99±0.01 | 0.85±0.03 | 31.46±0.23 |
| +1.00 | +21 | base | 12.72 | 4.06 | — | 0.78 | 29.60 |
| +2.00 | +42 | P | 14.46±0.74 | 4.48±0.16 | 0.70±0.03 | 0.47±0.01 | 73.71±0.17 |
| +2.00 | +42 | A | 14.59±0.29 | 4.68±0.33 | 0.84±0.02 | 0.61±0.06 | 74.29±0.24 |
| +2.00 | +42 | base | 13.76 | 3.52 | — | 0.66 | 72.60 |

Social specificity by dose (P−A on trust vs matched lottery): dose-41.73: trust 0.5337 / lottery 0.0159; dose-20.86: trust 1.4529 / lottery -0.1564; dose-10.43: trust 1.4046 / lottery -0.1365; dose-4.00: trust 1.2658 / lottery -0.1329; dose-2.00: trust 1.2825 / lottery -0.1439; dose-1.00: trust 1.3443 / lottery -0.1427; dose+0.00: trust 1.3122 / lottery -0.1423; dose+1.00: trust 1.3107 / lottery -0.1452; dose+2.00: trust 1.3542 / lottery -0.1467; dose+4.00: trust 1.4614 / lottery -0.1516; dose+10.43: trust 1.7225 / lottery -0.1511; dose+20.86: trust 1.8055 / lottery -0.1704; dose+41.73: trust -0.2736 / lottery -0.1391; dose-41.73: trust 1.5612 / lottery -0.0683; dose-20.86: trust 1.1177 / lottery -0.104; dose-10.43: trust 1.1445 / lottery -0.1146; dose+0.00: trust 1.3093 / lottery -0.1423; dose+10.43: trust 1.5589 / lottery -0.1733; dose+20.86: trust 1.8243 / lottery -0.1647; dose+41.73: trust 2.0373 / lottery -0.1703

**anger@L22-24** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
| -2.00 | -68 | P | 18.50±0.46 | 5.99±0.07 | 1.00±0.00 | 0.53±0.04 | -79.51±0.11 |
| -2.00 | -68 | A | 16.70±0.64 | 5.33±0.03 | 0.99±0.00 | 0.56±0.04 | -78.09±0.10 |
| -1.00 | -34 | P | 15.11±0.33 | 5.94±0.10 | 1.00±0.00 | 0.55±0.01 | -35.97±0.08 |
| -1.00 | -34 | A | 13.30±0.46 | 4.78±0.14 | 1.00±0.00 | 0.73±0.00 | -34.88±0.11 |
| -0.50 | -17 | P | 13.73±0.18 | 5.90±0.03 | 1.00±0.00 | 0.57±0.02 | -14.31±0.09 |
| -0.50 | -17 | A | 11.14±0.41 | 4.70±0.15 | 0.99±0.01 | 0.74±0.01 | -13.41±0.13 |
| +0.00 | +0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | 7.53±0.12 |
| +0.00 | +0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | 8.19±0.14 |
| +0.50 | +17 | P | 12.24±0.14 | 5.42±0.05 | 1.00±0.00 | 0.66±0.04 | 29.66±0.17 |
| +0.50 | +17 | A | 10.15±0.56 | 3.02±0.42 | 0.96±0.01 | 0.81±0.01 | 30.06±0.14 |
| +1.00 | +34 | P | 12.29±0.24 | 5.35±0.11 | 1.00±0.00 | 0.67±0.02 | 52.07±0.23 |
| +1.00 | +34 | A | 10.90±0.38 | 3.04±0.40 | 0.95±0.00 | 0.80±0.03 | 52.20±0.13 |
| +2.00 | +68 | P | 15.06±0.41 | 5.57±0.13 | 1.00±0.00 | 0.50±0.03 | 97.70±0.32 |
| +2.00 | +68 | A | 15.03±0.29 | 4.71±0.05 | 0.95±0.00 | 0.56±0.02 | 97.29±0.08 |

Social specificity by dose (P−A on trust vs matched lottery): dose-68.19: trust 0.7091 / lottery -0.0282; dose-34.10: trust 0.8543 / lottery -0.1808; dose-17.05: trust 0.9314 / lottery -0.1727; dose+0.00: trust 1.3122 / lottery -0.1423; dose+17.05: trust 2.1714 / lottery -0.1531; dose+34.10: trust 2.0937 / lottery -0.1319; dose+68.19: trust 0.395 / lottery -0.0601

**disgust@L22-24** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
| -2.00 | -50 | P | 14.84±0.28 | 6.48±0.02 | 1.00±0.00 | 0.83±0.02 | -94.85±0.27 |
| -2.00 | -50 | A | 14.04±0.12 | 5.74±0.20 | 0.96±0.00 | 0.84±0.02 | -93.56±0.06 |
| -1.00 | -25 | P | 13.34±0.25 | 6.15±0.08 | 1.00±0.00 | 0.74±0.02 | -51.03±0.23 |
| -1.00 | -25 | A | 9.58±0.06 | 4.44±0.08 | 0.96±0.00 | 0.84±0.02 | -49.92±0.10 |
| -0.50 | -12 | P | 12.69±0.21 | 5.93±0.07 | 1.00±0.00 | 0.70±0.03 | -29.35±0.23 |
| -0.50 | -12 | A | 8.80±0.09 | 3.95±0.07 | 0.96±0.01 | 0.82±0.02 | -28.33±0.10 |
| +0.00 | +0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | -7.51±0.23 |
| +0.00 | +0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | -6.59±0.11 |
| +0.50 | +12 | P | 12.59±0.19 | 5.44±0.10 | 1.00±0.00 | 0.58±0.04 | 14.79±0.23 |
| +0.50 | +12 | A | 10.91±0.82 | 4.14±0.26 | 0.99±0.00 | 0.78±0.02 | 15.61±0.13 |
| +1.00 | +25 | P | 12.51±0.25 | 5.15±0.07 | 1.00±0.00 | 0.64±0.02 | 37.75±0.24 |
| +1.00 | +25 | A | 10.48±0.68 | 4.33±0.46 | 1.00±0.00 | 0.81±0.02 | 38.40±0.16 |
| +2.00 | +50 | P | 16.80±0.43 | 4.86±0.11 | 0.64±0.09 | 0.38±0.03 | 85.58±0.32 |
| +2.00 | +50 | A | 14.46±0.30 | 5.43±0.13 | 0.94±0.00 | 0.46±0.04 | 85.64±0.26 |

Social specificity by dose (P−A on trust vs matched lottery): dose-49.56: trust 0.7049 / lottery -0.0148; dose-24.78: trust 1.5542 / lottery -0.0953; dose-12.39: trust 1.7522 / lottery -0.1204; dose+0.00: trust 1.3122 / lottery -0.1423; dose+12.39: trust 0.952 / lottery -0.1952; dose+24.78: trust 0.6353 / lottery -0.171; dose+49.56: trust -0.654 / lottery -0.076

**disgust@L60** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
|  | -9 | P | 12.69±0.17 | 5.71±0.03 | 1.00±0.00 | 0.61±0.04 | -500.19±2.45 |
|  | -9 | A | 10.06±0.35 | 4.11±0.13 | 0.97±0.01 | 0.76±0.01 | -469.60±3.19 |
|  | -9 | base | 11.41 | 4.14 | — | 0.76 | -484.87 |
|  | -7 | P | 12.73±0.17 | 5.71±0.03 | 1.00±0.00 | 0.61±0.04 | -432.10±2.46 |
|  | -7 | A | 10.13±0.36 | 4.11±0.12 | 0.97±0.01 | 0.76±0.01 | -401.50±3.19 |
|  | -7 | base | 11.30 | 4.09 | — | 0.76 | -416.51 |
|  | -4 | P | 12.76±0.17 | 5.73±0.03 | 1.00±0.00 | 0.63±0.04 | -349.26±2.45 |
|  | -4 | A | 10.13±0.35 | 4.10±0.12 | 0.97±0.01 | 0.77±0.01 | -318.67±3.18 |
|  | -4 | base | 11.19 | 4.12 | — | 0.77 | -333.90 |
|  | -2 | P | 12.77±0.17 | 5.72±0.02 | 1.00±0.00 | 0.63±0.04 | -288.06±2.45 |
|  | -2 | A | 10.09±0.34 | 4.10±0.12 | 0.97±0.01 | 0.77±0.01 | -257.45±3.19 |
|  | -2 | base | 11.14 | 4.12 | — | 0.78 | -272.61 |
|  | -1 | P | 12.77±0.20 | 5.73±0.02 | 1.00±0.00 | 0.64±0.04 | -257.94±2.44 |
|  | -1 | A | 10.14±0.35 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | -227.36±3.19 |
|  | -1 | base | 11.10 | 4.14 | — | 0.78 | -242.23 |
|  | +0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | -226.72±2.45 |
|  | +0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | -196.11±3.19 |
|  | +0 | base | 11.05 | 4.11 | — | 0.78 | -211.20 |
|  | +1 | P | 12.80±0.20 | 5.74±0.03 | 1.00±0.00 | 0.64±0.04 | -195.49±2.45 |
|  | +1 | A | 10.16±0.34 | 4.05±0.12 | 0.97±0.01 | 0.78±0.01 | -164.90±3.15 |
|  | +1 | base | 10.96 | 4.15 | — | 0.79 | -180.23 |
|  | +2 | P | 12.77±0.20 | 5.73±0.02 | 1.00±0.00 | 0.64±0.05 | -165.37±2.44 |
|  | +2 | A | 10.16±0.31 | 4.09±0.12 | 0.97±0.01 | 0.78±0.01 | -134.79±3.19 |
|  | +2 | base | 10.95 | 4.13 | — | 0.79 | -149.87 |
|  | +4 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | -104.17±2.45 |
|  | +4 | A | 10.15±0.32 | 4.07±0.11 | 0.97±0.01 | 0.79±0.01 | -73.49±3.25 |
|  | +4 | base | 10.94 | 4.11 | — | 0.79 | -88.15 |
|  | +7 | P | 12.78±0.20 | 5.73±0.03 | 1.00±0.00 | 0.65±0.04 | -21.34±2.44 |
|  | +7 | A | 10.16±0.36 | 4.03±0.09 | 0.97±0.01 | 0.80±0.01 | 9.27±3.19 |
|  | +7 | base | 10.87 | 4.17 | — | 0.80 | -5.82 |
|  | +9 | P | 12.76±0.19 | 5.74±0.03 | 1.00±0.00 | 0.66±0.05 | 46.77±2.45 |
|  | +9 | A | 10.18±0.31 | 4.01±0.10 | 0.97±0.01 | 0.80±0.01 | 77.53±3.30 |
|  | +9 | base | 10.88 | 4.17 | — | 0.80 | 62.91 |

Social specificity by dose (P−A on trust vs matched lottery): dose-8.90: trust 1.2538 / lottery -0.1473; dose-6.68: trust 1.2478 / lottery -0.1486; dose-4.00: trust 1.2891 / lottery -0.1441; dose-2.00: trust 1.2827 / lottery -0.1447; dose-1.00: trust 1.3047 / lottery -0.1401; dose+0.00: trust 1.3122 / lottery -0.1423; dose+1.00: trust 1.3666 / lottery -0.1419; dose+2.00: trust 1.294 / lottery -0.146; dose+4.00: trust 1.3271 / lottery -0.1454; dose+6.68: trust 1.3494 / lottery -0.1427; dose+8.90: trust 1.3593 / lottery -0.1399

**fear@L22-24** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
| -2.00 | -391 | P | 13.28±0.37 | 6.24±0.02 | 1.00±0.00 | 0.63±0.02 | -97.43±0.10 |
| -2.00 | -391 | A | 11.47±0.43 | 5.34±0.19 | 0.95±0.00 | 0.74±0.02 | -96.89±0.23 |
| -1.00 | -195 | P | 12.05±0.12 | 6.02±0.04 | 1.00±0.00 | 0.69±0.03 | -54.22±0.08 |
| -1.00 | -195 | A | 9.52±0.49 | 3.29±0.25 | 0.95±0.00 | 0.80±0.01 | -53.89±0.19 |
| -0.50 | -98 | P | 12.29±0.19 | 5.79±0.05 | 1.00±0.00 | 0.66±0.03 | -32.86±0.07 |
| -0.50 | -98 | A | 9.84±0.55 | 3.22±0.24 | 0.95±0.00 | 0.79±0.01 | -32.65±0.19 |
| +0.00 | +0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | -11.30±0.09 |
| +0.00 | +0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | -11.19±0.18 |
| +0.50 | +98 | P | 13.48±0.12 | 5.72±0.08 | 1.00±0.00 | 0.66±0.04 | 10.92±0.14 |
| +0.50 | +98 | A | 10.53±0.49 | 4.66±0.16 | 1.00±0.00 | 0.84±0.01 | 10.99±0.18 |
| +1.00 | +195 | P | 14.27±0.27 | 5.61±0.12 | 1.00±0.00 | 0.76±0.03 | 34.26±0.22 |
| +1.00 | +195 | A | 12.62±0.48 | 5.28±0.23 | 1.00±0.00 | 0.91±0.01 | 34.27±0.25 |
| +2.00 | +391 | P | 17.70±0.47 | 5.79±0.08 | 0.94±0.02 | 0.67±0.02 | 84.68±0.41 |
| +2.00 | +391 | A | 17.85±0.37 | 5.66±0.16 | 0.99±0.00 | 0.85±0.03 | 84.30±0.61 |

Social specificity by dose (P−A on trust vs matched lottery): dose-390.52: trust 0.738 / lottery -0.1134; dose-195.26: trust 2.4404 / lottery -0.1085; dose-97.63: trust 2.2239 / lottery -0.1281; dose+0.00: trust 1.3122 / lottery -0.1423; dose+97.63: trust 0.8582 / lottery -0.1746; dose+195.26: trust 0.2739 / lottery -0.149; dose+390.52: trust 0.1827 / lottery -0.1833

**joy@L22-24** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
| -2.00 | +45 | P | 15.03±0.42 | 4.48±0.13 | 0.87±0.01 | 0.44±0.01 | -78.63±0.16 |
| -2.00 | +45 | A | 15.15±0.09 | 4.08±0.48 | 0.91±0.01 | 0.66±0.03 | -79.16±0.29 |
| -1.00 | +22 | P | 12.99±0.34 | 4.86±0.12 | 1.00±0.00 | 0.71±0.01 | -36.49±0.09 |
| -1.00 | +22 | A | 10.75±0.46 | 2.90±0.53 | 0.99±0.01 | 0.84±0.02 | -37.16±0.25 |
| -0.50 | +11 | P | 12.62±0.33 | 5.22±0.08 | 1.00±0.00 | 0.66±0.05 | -14.92±0.05 |
| -0.50 | +11 | A | 10.11±0.38 | 3.33±0.21 | 0.97±0.01 | 0.83±0.01 | -15.71±0.24 |
| +0.00 | -0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | 7.03±0.04 |
| +0.00 | -0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | 6.02±0.22 |
| +0.50 | -11 | P | 12.77±0.23 | 6.03±0.02 | 1.00±0.00 | 0.65±0.02 | 29.14±0.06 |
| +0.50 | -11 | A | 10.40±0.64 | 4.29±0.17 | 0.97±0.01 | 0.76±0.03 | 27.82±0.22 |
| +1.00 | -22 | P | 13.54±0.26 | 6.12±0.07 | 1.00±0.00 | 0.68±0.02 | 51.17±0.09 |
| +1.00 | -22 | A | 12.09±0.88 | 4.27±0.25 | 0.99±0.01 | 0.79±0.01 | 49.56±0.23 |
| +2.00 | -45 | P | 15.42±0.59 | 6.19±0.06 | 0.99±0.00 | 0.73±0.02 | 95.44±0.16 |
| +2.00 | -45 | A | 15.27±0.38 | 5.48±0.09 | 0.98±0.00 | 0.47±0.04 | 93.41±0.26 |

Social specificity by dose (P−A on trust vs matched lottery): dose-44.66: trust 0.7312 / lottery 0.2569; dose-22.33: trust 1.524 / lottery -0.1102; dose-11.17: trust 1.3864 / lottery -0.1096; dose-0.00: trust 1.3122 / lottery -0.1423; dose+11.17: trust 1.7729 / lottery -0.1637; dose+22.33: trust 2.1548 / lottery -0.1263; dose+44.66: trust 0.5303 / lottery -0.2247

**joy@L25** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
|  | +35 | P | 12.84±0.53 | 5.05±0.15 | 1.00±0.00 | 0.71±0.04 | -40.52±0.15 |
|  | +35 | A | 10.94±0.31 | 3.89±0.13 | 0.97±0.01 | 0.83±0.03 | -42.06±0.09 |
|  | +35 | base | 12.19 | 4.60 | — | 0.82 | -39.83 |
|  | +18 | P | 12.49±0.41 | 5.35±0.06 | 1.00±0.00 | 0.66±0.05 | -13.28±0.15 |
|  | +18 | A | 10.07±0.31 | 3.86±0.07 | 0.97±0.01 | 0.82±0.01 | -14.82±0.09 |
|  | +18 | base | 11.52 | 4.30 | — | 0.80 | -12.58 |
|  | +9 | P | 12.57±0.30 | 5.55±0.02 | 1.00±0.00 | 0.64±0.05 | 0.35±0.15 |
|  | +9 | A | 10.11±0.31 | 3.92±0.10 | 0.97±0.01 | 0.80±0.01 | -1.18±0.09 |
|  | +9 | base | 11.11 | 4.16 | — | 0.79 | 1.05 |
|  | +4 | P | 12.72±0.25 | 5.64±0.02 | 1.00±0.00 | 0.64±0.05 | 7.78±0.15 |
|  | +4 | A | 10.11±0.35 | 3.97±0.10 | 0.97±0.01 | 0.79±0.01 | 6.24±0.09 |
|  | +4 | base | 11.12 | 4.14 | — | 0.79 | 8.47 |
|  | +2 | P | 12.76±0.21 | 5.70±0.02 | 1.00±0.00 | 0.64±0.05 | 10.84±0.15 |
|  | +2 | A | 10.14±0.34 | 4.05±0.13 | 0.97±0.01 | 0.78±0.01 | 9.31±0.09 |
|  | +2 | base | 11.09 | 4.08 | — | 0.79 | 11.54 |
|  | +1 | P | 12.78±0.19 | 5.72±0.03 | 1.00±0.00 | 0.64±0.05 | 12.42±0.15 |
|  | +1 | A | 10.15±0.33 | 4.05±0.11 | 0.97±0.01 | 0.78±0.01 | 10.88±0.09 |
|  | +1 | base | 11.09 | 4.10 | — | 0.79 | 13.11 |
|  | -0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | 13.95±0.15 |
|  | -0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | 12.42±0.09 |
|  | -0 | base | 11.05 | 4.11 | — | 0.78 | 14.65 |
|  | -1 | P | 12.78±0.14 | 5.76±0.02 | 1.00±0.00 | 0.63±0.04 | 15.49±0.15 |
|  | -1 | A | 10.15±0.37 | 4.08±0.11 | 0.97±0.01 | 0.78±0.01 | 13.96±0.09 |
|  | -1 | base | 11.08 | 4.14 | — | 0.79 | 16.19 |
|  | -2 | P | 12.78±0.15 | 5.77±0.04 | 1.00±0.00 | 0.63±0.04 | 17.06±0.15 |
|  | -2 | A | 10.10±0.37 | 4.10±0.12 | 0.97±0.01 | 0.78±0.01 | 15.53±0.09 |
|  | -2 | base | 11.02 | 4.05 | — | 0.78 | 17.76 |
|  | -4 | P | 12.84±0.13 | 5.80±0.03 | 1.00±0.00 | 0.63±0.04 | 20.13±0.15 |
|  | -4 | A | 10.15±0.37 | 4.15±0.13 | 0.97±0.01 | 0.77±0.01 | 18.60±0.09 |
|  | -4 | base | 11.13 | 4.09 | — | 0.78 | 20.83 |
|  | -9 | P | 12.93±0.17 | 5.88±0.03 | 1.00±0.00 | 0.63±0.04 | 27.55±0.15 |
|  | -9 | A | 10.10±0.45 | 4.18±0.15 | 0.97±0.01 | 0.76±0.00 | 26.01±0.09 |
|  | -9 | base | 11.35 | 4.17 | — | 0.77 | 28.25 |
|  | -18 | P | 12.99±0.28 | 5.98±0.04 | 1.00±0.00 | 0.64±0.03 | 41.19±0.15 |
|  | -18 | A | 10.01±0.53 | 4.20±0.17 | 0.98±0.01 | 0.76±0.01 | 39.65±0.09 |
|  | -18 | base | 11.63 | 4.08 | — | 0.74 | 41.88 |
|  | -35 | P | 13.08±0.56 | 6.06±0.02 | 1.00±0.00 | 0.67±0.02 | 68.43±0.15 |
|  | -35 | A | 10.87±0.83 | 3.94±0.13 | 0.99±0.01 | 0.77±0.01 | 66.90±0.09 |
|  | -35 | base | 11.94 | 3.64 | — | 0.65 | 69.13 |

Social specificity by dose (P−A on trust vs matched lottery): dose-35.25: trust 1.8452 / lottery -0.0976; dose-17.63: trust 1.4813 / lottery -0.112; dose-8.81: trust 1.3812 / lottery -0.1328; dose-4.00: trust 1.315 / lottery -0.1401; dose-2.00: trust 1.3228 / lottery -0.1456; dose-1.00: trust 1.3539 / lottery -0.1486; dose-0.00: trust 1.3122 / lottery -0.1423; dose+1.00: trust 1.338 / lottery -0.1469; dose+2.00: trust 1.2977 / lottery -0.1459; dose+4.00: trust 1.3356 / lottery -0.1483; dose+8.81: trust 1.3294 / lottery -0.1534; dose+17.63: trust 1.2538 / lottery -0.1558; dose+35.25: trust 1.0752 / lottery -0.1222

**sadness@L22-24** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
| -2.00 | -1987 | P | 13.82±0.47 | 6.15±0.01 | 1.00±0.00 | 0.53±0.02 | -102.25±0.47 |
| -2.00 | -1987 | A | 13.43±0.09 | 5.87±0.12 | 1.00±0.00 | 0.77±0.01 | -101.81±0.15 |
| -1.00 | -993 | P | 12.59±0.10 | 5.80±0.06 | 1.00±0.00 | 0.60±0.03 | -58.88±0.47 |
| -1.00 | -993 | A | 10.90±0.19 | 4.69±0.10 | 0.99±0.00 | 0.81±0.02 | -58.57±0.07 |
| -0.50 | -497 | P | 12.60±0.11 | 5.76±0.04 | 1.00±0.00 | 0.59±0.04 | -37.22±0.48 |
| -0.50 | -497 | A | 10.57±0.33 | 4.51±0.09 | 0.98±0.01 | 0.78±0.01 | -37.04±0.04 |
| +0.00 | +0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | -15.61±0.49 |
| +0.00 | +0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | -15.59±0.03 |
| +0.50 | +497 | P | 12.99±0.32 | 5.60±0.06 | 1.00±0.00 | 0.71±0.04 | 6.29±0.53 |
| +0.50 | +497 | A | 11.18±0.36 | 3.06±0.13 | 0.97±0.01 | 0.84±0.00 | 6.14±0.04 |
| +1.00 | +993 | P | 14.15±0.32 | 5.60±0.10 | 1.00±0.00 | 0.76±0.03 | 28.85±0.57 |
| +1.00 | +993 | A | 13.53±0.52 | 2.70±0.35 | 0.97±0.01 | 0.87±0.01 | 28.55±0.06 |
| +2.00 | +1987 | P | 19.13±0.90 | 4.84±0.37 | 0.99±0.00 | 0.44±0.05 | 76.10±0.58 |
| +2.00 | +1987 | A | 16.17±0.25 | 5.36±0.15 | 0.99±0.00 | 0.36±0.01 | 75.56±0.10 |

Social specificity by dose (P−A on trust vs matched lottery): dose-1986.54: trust 0.0519 / lottery -0.2397; dose-993.27: trust 0.6669 / lottery -0.2109; dose-496.63: trust 0.7806 / lottery -0.1872; dose+0.00: trust 1.3122 / lottery -0.1423; dose+496.63: trust 2.4291 / lottery -0.1356; dose+993.27: trust 2.9618 / lottery -0.1095; dose+1986.54: trust -0.3983 / lottery 0.0804

**surprise@L22-24** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
| -2.00 | +114 | P | 15.99±0.73 | 4.97±0.17 | 0.99±0.00 | 0.70±0.01 | -73.02±0.25 |
| -2.00 | +114 | A | 14.54±0.61 | 5.68±0.10 | 0.99±0.00 | 0.84±0.03 | -73.22±0.11 |
| -1.00 | +57 | P | 12.36±0.18 | 5.94±0.11 | 1.00±0.00 | 0.73±0.07 | -29.35±0.27 |
| -1.00 | +57 | A | 12.04±0.29 | 4.55±0.57 | 1.00±0.00 | 0.86±0.03 | -29.65±0.10 |
| -0.50 | +28 | P | 12.75±0.07 | 5.78±0.12 | 1.00±0.00 | 0.66±0.06 | -7.73±0.28 |
| -0.50 | +28 | A | 11.02±0.53 | 4.22±0.27 | 0.99±0.01 | 0.79±0.02 | -8.07±0.09 |
| +0.00 | -0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | 13.91±0.27 |
| +0.00 | -0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | 13.51±0.07 |
| +0.50 | -28 | P | 12.74±0.26 | 5.82±0.04 | 1.00±0.00 | 0.66±0.03 | 35.66±0.27 |
| +0.50 | -28 | A | 8.74±0.07 | 3.67±0.05 | 0.96±0.01 | 0.78±0.01 | 35.17±0.06 |
| +1.00 | -57 | P | 12.76±0.43 | 6.08±0.05 | 1.00±0.00 | 0.67±0.01 | 57.51±0.26 |
| +1.00 | -57 | A | 7.82±0.19 | 3.88±0.21 | 0.96±0.01 | 0.78±0.01 | 56.90±0.06 |
| +2.00 | -114 | P | 18.72±0.35 | 6.13±0.11 | 1.00±0.00 | 0.57±0.04 | 101.81±0.24 |
| +2.00 | -114 | A | 18.46±0.40 | 5.82±0.11 | 0.97±0.00 | 0.65±0.02 | 100.99±0.02 |

Social specificity by dose (P−A on trust vs matched lottery): dose-113.94: trust 0.1257 / lottery -0.078; dose-56.97: trust 2.1078 / lottery -0.1137; dose-28.48: trust 1.9118 / lottery -0.1244; dose-0.00: trust 1.3122 / lottery -0.1423; dose+28.48: trust 1.2474 / lottery -0.1326; dose+56.97: trust 1.0813 / lottery -0.1257; dose+113.94: trust -0.5266 / lottery -0.1375

**valence@L22-24|gen** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
| -2.00 | -42 | P | 12.13±0.12 | 5.48±0.02 | 1.00±0.00 | 0.64±0.06 | -99.30±0.14 |
| -2.00 | -42 | A | 9.66±0.30 | 3.59±0.26 | 0.97±0.01 | 0.70±0.02 | -97.64±0.23 |
| -1.00 | -21 | P | 12.34±0.10 | 5.61±0.04 | 1.00±0.00 | 0.62±0.05 | -56.57±0.14 |
| -1.00 | -21 | A | 9.72±0.39 | 4.20±0.10 | 0.97±0.01 | 0.73±0.01 | -55.09±0.22 |
| -0.50 | -10 | P | 12.53±0.06 | 5.72±0.03 | 1.00±0.00 | 0.64±0.04 | -35.09±0.14 |
| -0.50 | -10 | A | 9.86±0.32 | 4.27±0.05 | 0.97±0.01 | 0.75±0.01 | -33.79±0.21 |
| +0.00 | +0 | P | 12.77±0.19 | 5.73±0.03 | 1.00±0.00 | 0.64±0.04 | -13.49±0.14 |
| +0.00 | +0 | A | 10.13±0.34 | 4.09±0.12 | 0.97±0.01 | 0.78±0.01 | -12.40±0.20 |
| +0.50 | +10 | P | 12.77±0.25 | 5.62±0.02 | 1.00±0.00 | 0.63±0.05 | 8.23±0.13 |
| +0.50 | +10 | A | 10.46±0.29 | 3.73±0.23 | 0.97±0.01 | 0.81±0.01 | 9.12±0.18 |
| +1.00 | +21 | P | 12.71±0.15 | 5.56±0.01 | 1.00±0.00 | 0.66±0.05 | 29.95±0.13 |
| +1.00 | +21 | A | 10.53±0.29 | 3.49±0.40 | 0.97±0.01 | 0.82±0.01 | 30.70±0.17 |
| +2.00 | +42 | P | 12.41±0.20 | 5.56±0.03 | 1.00±0.00 | 0.68±0.05 | 73.33±0.14 |
| +2.00 | +42 | A | 9.88±0.40 | 3.27±0.28 | 0.97±0.01 | 0.85±0.02 | 73.83±0.18 |

Social specificity by dose (P−A on trust vs matched lottery): dose-41.73: trust 1.5612 / lottery -0.0683; dose-20.86: trust 1.1177 / lottery -0.104; dose-10.43: trust 1.1445 / lottery -0.1146; dose+0.00: trust 1.3093 / lottery -0.1423; dose+10.43: trust 1.5589 / lottery -0.1733; dose+20.86: trust 1.8243 / lottery -0.1647; dose+41.73: trust 2.0373 / lottery -0.1703

**valence@L32-34** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
| -2.00 | -951 | P | 13.82±0.65 | 1.89±0.28 | 1.00±0.00 | 0.55±0.04 | -105.16±0.38 |
| -2.00 | -951 | A | 10.51±0.57 | 1.84±0.34 | 0.91±0.01 | 0.46±0.12 | -104.96±0.67 |
| -1.00 | -475 | P | 12.30±0.30 | 5.84±0.06 | 1.00±0.00 | 0.64±0.02 | -61.29±0.38 |
| -1.00 | -475 | A | 8.56±0.26 | 3.19±0.14 | 0.95±0.00 | 0.69±0.04 | -61.16±0.67 |
| -0.50 | -238 | P | 12.42±0.16 | 5.88±0.03 | 1.00±0.00 | 0.63±0.03 | -39.33±0.38 |
| -0.50 | -238 | A | 9.25±0.22 | 3.56±0.13 | 0.96±0.01 | 0.69±0.02 | -39.25±0.66 |
| +0.00 | +0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | -17.22±0.39 |
| +0.00 | +0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | -17.18±0.65 |
| +0.50 | +238 | P | 12.92±0.45 | 5.28±0.10 | 1.00±0.00 | 0.69±0.04 | 4.97±0.43 |
| +0.50 | +238 | A | 12.16±1.03 | 4.63±0.11 | 0.99±0.01 | 0.84±0.01 | 4.95±0.64 |
| +1.00 | +475 | P | 12.57±0.67 | 4.43±0.22 | 1.00±0.00 | 0.51±0.06 | 27.08±0.49 |
| +1.00 | +475 | A | 13.38±0.98 | 4.76±0.15 | 1.00±0.00 | 0.83±0.02 | 26.99±0.61 |
| +2.00 | +951 | P | 14.81±0.66 | 4.06±0.17 | 0.88±0.01 | 0.57±0.04 | 71.05±0.65 |
| +2.00 | +951 | A | 14.53±0.32 | 5.39±0.13 | 0.93±0.01 | 0.77±0.03 | 70.73±0.60 |

Social specificity by dose (P−A on trust vs matched lottery): dose-950.89: trust 0.2646 / lottery 0.0973; dose-475.44: trust 2.3412 / lottery -0.0471; dose-237.72: trust 1.9771 / lottery -0.0685; dose+0.00: trust 1.3122 / lottery -0.1423; dose+237.72: trust 0.3974 / lottery -0.1519; dose+475.44: trust -0.4716 / lottery -0.3211; dose+950.89: trust -1.1281 / lottery -0.1995

**valence@L46-48** (single direction, robustness) — expected return ($ of 30), amount sent ($ of 10), own-world endorsement (P of world letter), lottery P(take), projection:

| α (method) | dose (gap units) | arm | exp. return | amount | own-world | lottery | projection |
|---|---|---|---|---|---|---|---|
| -2.00 | -22 | P | 10.37±0.51 | 6.10±0.02 | 0.99±0.00 | 0.79±0.03 | -130.02±0.46 |
| -2.00 | -22 | A | 8.34±0.15 | 5.93±0.02 | 0.95±0.01 | 0.78±0.02 | -128.59±1.30 |
| -1.00 | -11 | P | 12.55±0.35 | 6.50±0.05 | 1.00±0.00 | 0.77±0.03 | -67.81±0.37 |
| -1.00 | -11 | A | 11.02±0.11 | 6.17±0.04 | 0.97±0.01 | 0.78±0.01 | -65.54±1.32 |
| -0.50 | -6 | P | 13.10±0.24 | 6.39±0.10 | 1.00±0.00 | 0.74±0.04 | -36.27±0.32 |
| -0.50 | -6 | A | 11.13±0.15 | 5.52±0.09 | 0.97±0.01 | 0.78±0.00 | -33.54±1.31 |
| +0.00 | +0 | P | 12.77±0.19 | 5.74±0.02 | 1.00±0.00 | 0.64±0.04 | -4.21±0.28 |
| +0.00 | +0 | A | 10.13±0.34 | 4.08±0.12 | 0.97±0.01 | 0.78±0.01 | -1.31±1.30 |
| +0.50 | +6 | P | 12.15±0.25 | 4.92±0.15 | 1.00±0.00 | 0.58±0.05 | 27.95±0.25 |
| +0.50 | +6 | A | 10.52±0.47 | 4.24±0.18 | 0.97±0.02 | 0.86±0.01 | 30.57±1.28 |
| +1.00 | +11 | P | 13.26±0.47 | 6.21±0.09 | 1.00±0.00 | 0.78±0.04 | 59.64±0.19 |
| +1.00 | +11 | A | 12.99±0.98 | 6.26±0.32 | 0.97±0.02 | 0.94±0.01 | 61.77±1.27 |
| +2.00 | +22 | P | 13.68±0.81 | 7.42±0.02 | 1.00±0.00 | 0.94±0.02 | 121.44±0.10 |
| +2.00 | +22 | A | 13.48±1.23 | 7.56±0.08 | 0.99±0.01 | 0.97±0.00 | 122.81±1.29 |

Social specificity by dose (P−A on trust vs matched lottery): dose-22.11: trust 0.1003 / lottery 0.0122; dose-11.05: trust 0.2393 / lottery -0.0105; dose-5.53: trust 0.6219 / lottery -0.0459; dose+0.00: trust 1.3122 / lottery -0.1423; dose+5.53: trust 0.2403 / lottery -0.2764; dose+11.05: trust -0.4697 / lottery -0.157; dose+22.11: trust -0.1629 / lottery -0.037

## Finding 3 — this is not global persona drift (Exp 2, Assistant-Axis capping)

*Cap every condition inside the normal Assistant range (released Qwen3-32B `layers_46:54-p0.25`) and rerun belief + trust.*

| under capping… | it means |
|---|---|
| P/A belief difference disappears | may be different selected personas |
| belief persists, behavior difference disappears | stable world representation; drift controls expression |
| both persist | not explained by global drift from the Assistant persona |

| readout | P natural → capped | A natural → capped |
|---|---|---|
| own-world endorsement | 1.000±0.000 → 1.000±0.000 | 0.970±0.014 → 0.969±0.014 |
| expected return | 12.789±0.227 → 12.498±0.258 | 9.879±0.353 → 9.950±0.164 |
| amount sent | 5.776±0.036 → 5.548±0.018 | 4.010±0.109 → 3.285±0.156 |
| lottery P(take) | 0.625±0.040 → 0.633±0.042 | 0.776±0.012 → 0.796±0.013 |
| axis proj. (pre-response) | -18.847±0.289 → -18.847±0.289 | -20.199±0.520 → -20.199±0.520 |

Cap activation (fraction of tokens where the cap intervened):

| condition | cap active: last prompt token | cap active: generated tokens |
|---|---|---|
| base | 0.56 | 0.35 |
| P | 0.63 | 0.37 |
| S | 0.72 | 0.39 |
| N | 0.68 | 0.37 |
| A | 0.75 | 0.43 |
| MIX | 0.66 | 0.40 |
| REPLAY | 0.60 | 0.32 |

## Controls that the LoRA setup requires

- **Vectors still mean the same thing in every adapter:** held-out probe AUC of the negative-affect composite @L22–24 directions is 0.802 on base and 0.790–0.799 across the 18 adapters (no arm-specific collapse).

- **Content, not generic finetuning:** on the negative-affect composite @L22–24, N (-12.964±0.105) sits near base (-14.178) and REPLAY (-14.390±0.131) moves away from A (-12.394±0.195), not toward it; P = -13.485±0.137.

- **Adapter-strength dose response** (λ scales the LoRA update; λ=0 is base):

| arm | λ | own-world | negative-affect composite @L22–24 | exp. return | amount |
|---|---|---|---|---|---|
| A | 0.00 | 0.10±0.00 | -14.18±0.00 | 10.95±0.00 | 3.99±0.00 |
| A | 0.50 | 0.50±0.01 | -13.42±0.07 | 9.47±0.13 | 4.11±0.13 |
| A | 1.00 | 0.97±0.01 | -12.39±0.20 | 9.88±0.35 | 4.01±0.11 |
| P | 0.00 | 0.12±0.00 | -14.18±0.00 | 10.95±0.00 | 3.99±0.00 |
| P | 0.50 | 0.71±0.01 | -13.80±0.05 | 10.40±0.19 | 4.94±0.12 |
| P | 1.00 | 1.00±0.00 | -13.48±0.14 | 12.79±0.23 | 5.78±0.04 |

- **Social, not generic risk:** natural P−A gap on trust = 1.379 $ vs on the matched lottery = -0.1504 (P(take)).

## The composite in detail

Unit composite of the six base-derived directions (anger + disgust + fear + sadness − joy; surprise excluded as valence-neutral), averaged over the extraction method's own recommended layers [22, 23, 24] — pre-declared, no argmax over emotions or layers.

- trust-action prompts: d(A−P) half-1 = **3.614** (seed-consistent: True), half-2 = **3.566** (True); raw gap 1.095 / 1.087.
- conditions (seed mean ± SEM): base -14.178, P -13.485±0.137, S -13.432±0.102, N -12.964±0.105, A -12.394±0.195, MIX -12.480±0.157, REPLAY -14.390±0.131.
- same composite on lottery prompts: d half-1 = 1.496, half-2 = 1.524.
- same composite on belief prompts (no halves): P -15.195±0.380, A -14.514±0.203, base -16.148.

## Reviewer checks

- **How many directions are there really? cosine among the six emotion vectors at L22:**

|  | anger | disgust | fear | joy | sadness | surprise |
|---|---|---|---|---|---|---|
| anger | 1.0 | 0.19 | -0.32 | -0.36 | -0.25 | -0.09 |
| disgust | 0.19 | 1.0 | 0.11 | -0.43 | -0.24 | -0.53 |
| fear | -0.32 | 0.11 | 1.0 | -0.36 | 0.03 | -0.35 |
| joy | -0.36 | -0.43 | -0.36 | 1.0 | -0.02 | -0.02 |
| sadness | -0.25 | -0.24 | 0.03 | -0.02 | 1.0 | -0.27 |
| surprise | -0.09 | -0.53 | -0.35 | -0.02 | -0.27 | 1.0 |

- **How many directions are there really? cosine among the six emotion vectors at L23:**

|  | anger | disgust | fear | joy | sadness | surprise |
|---|---|---|---|---|---|---|
| anger | 1.0 | 0.18 | -0.33 | -0.35 | -0.26 | -0.08 |
| disgust | 0.18 | 1.0 | 0.11 | -0.43 | -0.24 | -0.52 |
| fear | -0.33 | 0.11 | 1.0 | -0.37 | 0.02 | -0.35 |
| joy | -0.35 | -0.43 | -0.37 | 1.0 | -0.01 | -0.03 |
| sadness | -0.26 | -0.24 | 0.02 | -0.01 | 1.0 | -0.26 |
| surprise | -0.08 | -0.52 | -0.35 | -0.03 | -0.26 | 1.0 |

- **How many directions are there really? cosine among the six emotion vectors at L24:**

|  | anger | disgust | fear | joy | sadness | surprise |
|---|---|---|---|---|---|---|
| anger | 1.0 | 0.18 | -0.33 | -0.34 | -0.25 | -0.1 |
| disgust | 0.18 | 1.0 | 0.11 | -0.43 | -0.23 | -0.52 |
| fear | -0.33 | 0.11 | 1.0 | -0.37 | 0.02 | -0.35 |
| joy | -0.34 | -0.43 | -0.37 | 1.0 | -0.02 | -0.02 |
| sadness | -0.25 | -0.23 | 0.02 | -0.02 | 1.0 | -0.27 |
| surprise | -0.1 | -0.52 | -0.35 | -0.02 | -0.27 | 1.0 |

- **How many directions are there really? cosine among the six emotion vectors at L25:**

|  | anger | disgust | fear | joy | sadness | surprise |
|---|---|---|---|---|---|---|
| anger | 1.0 | 0.17 | -0.34 | -0.33 | -0.27 | -0.08 |
| disgust | 0.17 | 1.0 | 0.13 | -0.43 | -0.2 | -0.54 |
| fear | -0.34 | 0.13 | 1.0 | -0.38 | 0.04 | -0.35 |
| joy | -0.33 | -0.43 | -0.38 | 1.0 | -0.03 | -0.02 |
| sadness | -0.27 | -0.2 | 0.04 | -0.03 | 1.0 | -0.29 |
| surprise | -0.08 | -0.54 | -0.35 | -0.02 | -0.29 | 1.0 |

- **How many directions are there really? cosine among the six emotion vectors at L60:**

|  | anger | disgust | fear | joy | sadness | surprise |
|---|---|---|---|---|---|---|
| anger | 1.0 | 0.26 | -0.29 | -0.45 | -0.31 | -0.12 |
| disgust | 0.26 | 1.0 | 0.13 | -0.49 | -0.12 | -0.53 |
| fear | -0.29 | 0.13 | 1.0 | -0.33 | 0.12 | -0.3 |
| joy | -0.45 | -0.49 | -0.33 | 1.0 | 0.03 | -0.02 |
| sadness | -0.31 | -0.12 | 0.12 | 0.03 | 1.0 | -0.42 |
| surprise | -0.12 | -0.53 | -0.3 | -0.02 | -0.42 | 1.0 |

- **Is the affect gap about the person in front of the model?** A−P projection gap on the negative-affect composite @L22–24 by prompt context (seed means):

| context | P | A | gap A−P | base |
|---|---|---|---|---|
| trust partner (perception) | -13.56 | -12.96 | 0.6 | -14.81 |
| trust partner (action) | -13.48 | -12.39 | 1.09 | -14.18 |
| matched lottery | -14.07 | -13.53 | 0.54 | -15.44 |
| world-belief MCQ | -15.2 | -14.51 | 0.68 | -16.15 |

- **By evidence level** (A−P on trust prompts; the prior should matter most with none/ambiguous evidence):

| evidence | amount | expected_return | projection |
|---|---|---|---|
| none | -1.02 | -5.55 | 0.81 |
| ambiguous | -1.74 | -1.7 | 1.15 |
| decisive | -2.54 | -1.48 | 1.32 |

- **Candidate-scored amount vs greedy-decoded amount:** r = 0.698, mean |Δ| = 2.386 $, greedy parse rate 0.982 over 1520 contexts.

- **Coherence under steering** (unnormalized next-token mass on the answer letters; parse rate of the greedy amount) — behaviour is interpreted only where these hold:

| direction | α raw | arm | letter mass | amount parse |
|---|---|---|---|---|
| anger@L22-24 | -45.690 | A | 0.961 | 1.0 |
| anger@L22-24 | -45.690 | P | 0.982 | 1.0 |
| anger@L22-24 | -22.845 | A | 0.988 | 1.0 |
| anger@L22-24 | -22.845 | P | 0.992 | 1.0 |
| anger@L22-24 | -11.423 | A | 0.989 | 1.0 |
| anger@L22-24 | -11.423 | P | 0.996 | 1.0 |
| anger@L22-24 | +0.000 | A | 0.99 | 1.0 |
| anger@L22-24 | +0.000 | P | 0.997 | 1.0 |
| anger@L22-24 | +11.423 | A | 0.992 | 1.0 |
| anger@L22-24 | +11.423 | P | 0.998 | 1.0 |
| anger@L22-24 | +22.845 | A | 0.994 | 1.0 |
| anger@L22-24 | +22.845 | P | 0.997 | 1.0 |
| anger@L22-24 | +45.690 | A | 0.984 | 1.0 |
| anger@L22-24 | +45.690 | P | 0.987 | 1.0 |
| disgust@L22-24 | -45.690 | A | 0.975 | 1.0 |
| disgust@L22-24 | -45.690 | P | 0.992 | 1.0 |
| disgust@L22-24 | -22.845 | A | 0.991 | 1.0 |
| disgust@L22-24 | -22.845 | P | 0.996 | 1.0 |
| disgust@L22-24 | -11.423 | A | 0.991 | 1.0 |
| disgust@L22-24 | -11.423 | P | 0.997 | 1.0 |
| disgust@L22-24 | +0.000 | A | 0.99 | 1.0 |
| disgust@L22-24 | +0.000 | P | 0.997 | 1.0 |
| disgust@L22-24 | +11.423 | A | 0.99 | 1.0 |
| disgust@L22-24 | +11.423 | P | 0.996 | 1.0 |
| disgust@L22-24 | +22.845 | A | 0.989 | 1.0 |
| disgust@L22-24 | +22.845 | P | 0.995 | 1.0 |
| disgust@L22-24 | +45.690 | A | 0.834 | 1.0 |
| disgust@L22-24 | +45.690 | P | 0.928 | 1.0 |
| disgust@L60 | -2.000 | A | 0.983 | 1.0 |
| disgust@L60 | -2.000 | P | 0.993 | 1.0 |
| disgust@L60 | -2.000 | base | 0.992 | 0.705 |
| disgust@L60 | -1.500 | A | 0.985 | 1.0 |
| disgust@L60 | -1.500 | P | 0.995 | 1.0 |
| disgust@L60 | -1.500 | base | 0.994 | 0.614 |
| disgust@L60 | -0.898 | A | — | 1.0 |
| disgust@L60 | -0.898 | P | — | 1.0 |
| disgust@L60 | -0.898 | base | — | 0.682 |
| disgust@L60 | -0.449 | A | — | 1.0 |
| disgust@L60 | -0.449 | P | — | 1.0 |
| disgust@L60 | -0.449 | base | — | 0.659 |
| disgust@L60 | -0.225 | A | — | 1.0 |
| disgust@L60 | -0.225 | P | — | 1.0 |
| disgust@L60 | -0.225 | base | — | 0.682 |
| disgust@L60 | +0.000 | A | — | 1.0 |
| disgust@L60 | +0.000 | P | — | 1.0 |
| disgust@L60 | +0.000 | base | — | 0.682 |
| disgust@L60 | +0.225 | A | — | 1.0 |
| disgust@L60 | +0.225 | P | — | 1.0 |
| disgust@L60 | +0.225 | base | — | 0.659 |
| disgust@L60 | +0.449 | A | — | 1.0 |
| disgust@L60 | +0.449 | P | — | 1.0 |
| disgust@L60 | +0.449 | base | — | 0.659 |
| disgust@L60 | +0.898 | A | — | 1.0 |
| disgust@L60 | +0.898 | P | — | 1.0 |
| disgust@L60 | +0.898 | base | — | 0.659 |
| disgust@L60 | +1.500 | A | 0.993 | 1.0 |
| disgust@L60 | +1.500 | P | 0.998 | 1.0 |
| disgust@L60 | +1.500 | base | 0.998 | 0.659 |
| disgust@L60 | +2.000 | A | 0.993 | 1.0 |
| disgust@L60 | +2.000 | P | 0.999 | 1.0 |
| disgust@L60 | +2.000 | base | 0.999 | 0.636 |
| fear@L22-24 | -45.690 | A | 0.988 | 1.0 |
| fear@L22-24 | -45.690 | P | 0.995 | 1.0 |
| fear@L22-24 | -22.845 | A | 0.993 | 1.0 |
| fear@L22-24 | -22.845 | P | 0.997 | 1.0 |
| fear@L22-24 | -11.423 | A | 0.992 | 1.0 |
| fear@L22-24 | -11.423 | P | 0.997 | 1.0 |
| fear@L22-24 | +0.000 | A | 0.99 | 1.0 |
| fear@L22-24 | +0.000 | P | 0.997 | 1.0 |
| fear@L22-24 | +11.423 | A | 0.989 | 1.0 |
| fear@L22-24 | +11.423 | P | 0.996 | 1.0 |
| fear@L22-24 | +22.845 | A | 0.987 | 1.0 |
| fear@L22-24 | +22.845 | P | 0.99 | 1.0 |
| fear@L22-24 | +45.690 | A | 0.938 | 1.0 |
| fear@L22-24 | +45.690 | P | 0.947 | 1.0 |
| joy@L22-24 | -45.690 | A | 0.972 | 1.0 |
| joy@L22-24 | -45.690 | P | 0.98 | 1.0 |
| joy@L22-24 | -22.845 | A | 0.991 | 1.0 |
| joy@L22-24 | -22.845 | P | 0.996 | 1.0 |
| joy@L22-24 | -11.423 | A | 0.991 | 1.0 |
| joy@L22-24 | -11.423 | P | 0.996 | 1.0 |
| joy@L22-24 | +0.000 | A | 0.99 | 1.0 |
| joy@L22-24 | +0.000 | P | 0.997 | 1.0 |
| joy@L22-24 | +11.423 | A | 0.99 | 1.0 |
| joy@L22-24 | +11.423 | P | 0.996 | 1.0 |
| joy@L22-24 | +22.845 | A | 0.991 | 1.0 |
| joy@L22-24 | +22.845 | P | 0.995 | 1.0 |
| joy@L22-24 | +45.690 | A | 0.988 | 1.0 |
| joy@L22-24 | +45.690 | P | 0.994 | 1.0 |
| joy@L25 | -2.000 | A | 0.991 | 1.0 |
| joy@L25 | -2.000 | P | 0.996 | 1.0 |
| joy@L25 | -2.000 | base | 0.997 | 1.0 |
| joy@L25 | -1.000 | A | 0.991 | 1.0 |
| joy@L25 | -1.000 | P | 0.997 | 1.0 |
| joy@L25 | -1.000 | base | 0.995 | 1.0 |
| joy@L25 | -0.500 | A | 0.99 | 1.0 |
| joy@L25 | -0.500 | P | 0.997 | 1.0 |
| joy@L25 | -0.500 | base | 0.996 | 0.886 |
| joy@L25 | -0.227 | A | 0.991 | 1.0 |
| joy@L25 | -0.227 | P | 0.997 | 1.0 |
| joy@L25 | -0.227 | base | — | 0.727 |
| joy@L25 | -0.113 | A | 0.992 | 1.0 |
| joy@L25 | -0.113 | P | 0.997 | 1.0 |
| joy@L25 | -0.113 | base | — | 0.659 |
| joy@L25 | -0.057 | A | 0.987 | 1.0 |
| joy@L25 | -0.057 | P | — | 1.0 |
| joy@L25 | -0.057 | base | 0.997 | 0.682 |
| joy@L25 | +0.000 | A | — | 1.0 |
| joy@L25 | +0.000 | P | 0.996 | 1.0 |
| joy@L25 | +0.000 | base | — | 0.682 |
| joy@L25 | +0.057 | A | 0.991 | 1.0 |
| joy@L25 | +0.057 | P | 0.997 | 1.0 |
| joy@L25 | +0.057 | base | — | 0.682 |
| joy@L25 | +0.113 | A | 0.992 | 1.0 |
| joy@L25 | +0.113 | P | 0.997 | 1.0 |
| joy@L25 | +0.113 | base | — | 0.682 |
| joy@L25 | +0.227 | A | 0.987 | 1.0 |
| joy@L25 | +0.227 | P | — | 1.0 |
| joy@L25 | +0.227 | base | 0.998 | 0.682 |
| joy@L25 | +0.500 | A | 0.99 | 1.0 |
| joy@L25 | +0.500 | P | 0.997 | 1.0 |
| joy@L25 | +0.500 | base | 0.998 | 0.727 |
| joy@L25 | +1.000 | A | 0.99 | 1.0 |
| joy@L25 | +1.000 | P | 0.997 | 1.0 |
| joy@L25 | +1.000 | base | 0.998 | 0.773 |
| joy@L25 | +2.000 | A | 0.991 | 1.0 |
| joy@L25 | +2.000 | P | 0.997 | 1.0 |
| joy@L25 | +2.000 | base | 0.998 | 0.909 |
| sadness@L22-24 | -45.690 | A | 0.986 | 1.0 |
| sadness@L22-24 | -45.690 | P | 0.993 | 1.0 |
| sadness@L22-24 | -22.845 | A | 0.99 | 1.0 |
| sadness@L22-24 | -22.845 | P | 0.997 | 1.0 |
| sadness@L22-24 | -11.423 | A | 0.99 | 1.0 |
| sadness@L22-24 | -11.423 | P | 0.997 | 1.0 |
| sadness@L22-24 | +0.000 | A | 0.99 | 1.0 |
| sadness@L22-24 | +0.000 | P | 0.997 | 1.0 |
| sadness@L22-24 | +11.423 | A | 0.99 | 1.0 |
| sadness@L22-24 | +11.423 | P | 0.997 | 1.0 |
| sadness@L22-24 | +22.845 | A | 0.991 | 1.0 |
| sadness@L22-24 | +22.845 | P | 0.996 | 1.0 |
| sadness@L22-24 | +45.690 | A | 0.964 | 1.0 |
| sadness@L22-24 | +45.690 | P | 0.976 | 1.0 |
| surprise@L22-24 | -45.690 | A | 0.96 | 1.0 |
| surprise@L22-24 | -45.690 | P | 0.98 | 1.0 |
| surprise@L22-24 | -22.845 | A | 0.989 | 1.0 |
| surprise@L22-24 | -22.845 | P | 0.997 | 1.0 |
| surprise@L22-24 | -11.423 | A | 0.99 | 1.0 |
| surprise@L22-24 | -11.423 | P | 0.997 | 1.0 |
| surprise@L22-24 | +0.000 | A | 0.99 | 1.0 |
| surprise@L22-24 | +0.000 | P | 0.997 | 1.0 |
| surprise@L22-24 | +11.423 | A | 0.99 | 1.0 |
| surprise@L22-24 | +11.423 | P | 0.996 | 1.0 |
| surprise@L22-24 | +22.845 | A | 0.987 | 1.0 |
| surprise@L22-24 | +22.845 | P | 0.993 | 1.0 |
| surprise@L22-24 | +45.690 | A | 0.952 | 1.0 |
| surprise@L22-24 | +45.690 | P | 0.975 | 1.0 |
| valence@L22-24 | -45.690 | A | 0.974 | 1.0 |
| valence@L22-24 | -45.690 | P | 0.989 | 1.0 |
| valence@L22-24 | -45.690 | base | 0.993 | 0.886 |
| valence@L22-24 | -22.845 | A | 0.992 | 1.0 |
| valence@L22-24 | -22.845 | P | 0.995 | 1.0 |
| valence@L22-24 | -22.845 | base | 0.996 | 0.818 |
| valence@L22-24 | -11.423 | A | 0.99 | 1.0 |
| valence@L22-24 | -11.423 | P | 0.996 | 1.0 |
| valence@L22-24 | -11.423 | base | 0.998 | 0.773 |
| valence@L22-24 | -4.380 | A | 0.99 | 1.0 |
| valence@L22-24 | -4.380 | P | 0.997 | 1.0 |
| valence@L22-24 | -4.380 | base | 0.998 | 0.727 |
| valence@L22-24 | -2.190 | A | 0.99 | 1.0 |
| valence@L22-24 | -2.190 | P | 0.997 | 1.0 |
| valence@L22-24 | -2.190 | base | 0.998 | 0.682 |
| valence@L22-24 | -1.095 | A | 0.99 | 1.0 |
| valence@L22-24 | -1.095 | P | 0.997 | 1.0 |
| valence@L22-24 | -1.095 | base | 0.997 | 0.682 |
| valence@L22-24 | +0.000 | A | 0.99 | 1.0 |
| valence@L22-24 | +0.000 | P | 0.997 | 1.0 |
| valence@L22-24 | +0.000 | base | 0.997 | 0.682 |
| valence@L22-24 | +1.095 | A | 0.99 | 1.0 |
| valence@L22-24 | +1.095 | P | 0.997 | 1.0 |
| valence@L22-24 | +1.095 | base | 0.997 | 0.682 |
| valence@L22-24 | +2.190 | A | 0.99 | 1.0 |
| valence@L22-24 | +2.190 | P | 0.997 | 1.0 |
| valence@L22-24 | +2.190 | base | 0.996 | 0.705 |
| valence@L22-24 | +4.380 | A | 0.99 | 1.0 |
| valence@L22-24 | +4.380 | P | 0.997 | 1.0 |
| valence@L22-24 | +4.380 | base | 0.996 | 0.841 |
| valence@L22-24 | +11.423 | A | 0.992 | 1.0 |
| valence@L22-24 | +11.423 | P | 0.997 | 1.0 |
| valence@L22-24 | +11.423 | base | 0.994 | 1.0 |
| valence@L22-24 | +22.845 | A | 0.992 | 1.0 |
| valence@L22-24 | +22.845 | P | 0.996 | 1.0 |
| valence@L22-24 | +22.845 | base | 0.99 | 1.0 |
| valence@L22-24 | +45.690 | A | 0.971 | 1.0 |
| valence@L22-24 | +45.690 | P | 0.98 | 1.0 |
| valence@L22-24 | +45.690 | base | 0.949 | 1.0 |
| valence@L22-24 | en | a-45.690 | 0.991 | 1.0 |
| valence@L22-24 | en | a-45.690 | 0.997 | 1.0 |
| valence@L22-24 | en | a-22.845 | 0.992 | 1.0 |
| valence@L22-24 | en | a-22.845 | 0.997 | 1.0 |
| valence@L22-24 | en | a-11.423 | 0.991 | 1.0 |
| valence@L22-24 | en | a-11.423 | 0.997 | 1.0 |
| valence@L22-24 | en | a+0.000 | 0.99 | 1.0 |
| valence@L22-24 | en | a+0.000 | 0.997 | 1.0 |
| valence@L22-24 | en | a+11.423 | 0.989 | 1.0 |
| valence@L22-24 | en | a+11.423 | 0.997 | 1.0 |
| valence@L22-24 | en | a+22.845 | 0.989 | 1.0 |
| valence@L22-24 | en | a+22.845 | 0.996 | 1.0 |
| valence@L22-24 | en | a+45.690 | 0.988 | 1.0 |
| valence@L22-24 | en | a+45.690 | 0.995 | 1.0 |
| valence@L32-34 | -47.544 | A | 0.985 | 1.0 |
| valence@L32-34 | -47.544 | P | 0.987 | 1.0 |
| valence@L32-34 | -23.772 | A | 0.992 | 1.0 |
| valence@L32-34 | -23.772 | P | 0.997 | 1.0 |
| valence@L32-34 | -11.886 | A | 0.991 | 1.0 |
| valence@L32-34 | -11.886 | P | 0.997 | 1.0 |
| valence@L32-34 | +0.000 | A | 0.99 | 1.0 |
| valence@L32-34 | +0.000 | P | 0.997 | 1.0 |
| valence@L32-34 | +11.886 | A | 0.989 | 1.0 |
| valence@L32-34 | +11.886 | P | 0.996 | 1.0 |
| valence@L32-34 | +23.772 | A | 0.989 | 1.0 |
| valence@L32-34 | +23.772 | P | 0.995 | 1.0 |
| valence@L32-34 | +47.544 | A | 0.983 | 1.0 |
| valence@L32-34 | +47.544 | P | 0.993 | 1.0 |
| valence@L46-48 | -64.859 | A | 0.97 | 1.0 |
| valence@L46-48 | -64.859 | P | 0.98 | 1.0 |
| valence@L46-48 | -32.429 | A | 0.99 | 1.0 |
| valence@L46-48 | -32.429 | P | 0.994 | 1.0 |
| valence@L46-48 | -16.215 | A | 0.991 | 1.0 |
| valence@L46-48 | -16.215 | P | 0.996 | 1.0 |
| valence@L46-48 | +0.000 | A | 0.99 | 1.0 |
| valence@L46-48 | +0.000 | P | 0.997 | 1.0 |
| valence@L46-48 | +16.215 | A | 0.989 | 1.0 |
| valence@L46-48 | +16.215 | P | 0.998 | 1.0 |
| valence@L46-48 | +32.429 | A | 0.987 | 1.0 |
| valence@L46-48 | +32.429 | P | 0.998 | 1.0 |
| valence@L46-48 | +64.859 | A | 0.984 | 1.0 |
| valence@L46-48 | +64.859 | P | 0.998 | 1.0 |

- **Verbatim greedy continuations under anger@L22-24 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -45.69 | 5.8257 | — | 7 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +45.69 | 5.7129 | — | 7 |
| trust.person.reciprocal.none.1 | A | -45.69 | 5.4113 | — | 7 |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | 5.1149 | — | 5 |
| trust.person.reciprocal.none.1 | P | -45.69 | — | 19.4171 | E / 30 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +45.69 | — | 14.898 | C / 15 |
| trust.person.reciprocal.none.1 | A | -45.69 | — | 18.0185 | E / 30 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | — | 14.0388 | B / 11 |

- **Verbatim greedy continuations under disgust@L22-24 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -45.69 | 6.4412 | — | 7 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +45.69 | 4.8991 | — | 7 |
| trust.person.reciprocal.none.1 | A | -45.69 | 5.6075 | — | I give 0 dollars. |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | 4.9125 | — | 10 |
| trust.person.reciprocal.none.1 | P | -45.69 | — | 15.2717 | C / $15 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +45.69 | — | 18.4833 | E / 30 |
| trust.person.reciprocal.none.1 | A | -45.69 | — | 12.9877 | A / $3 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | — | 15.7192 | C |

- **Verbatim greedy continuations under disgust@L60 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -2.00 | 5.4985 | — | 7 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +2.00 | 5.4235 | — | 7 |
| trust.person.reciprocal.none.1 | A | -2.00 | 4.5858 | — | I give 0 dollars. |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +2.00 | 4.2795 | — | 0 |
| trust.person.reciprocal.none.1 | P | -2.00 | — | 12.5254 | B / $9 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +2.00 | — | 12.5567 | B / 15 |
| trust.person.reciprocal.none.1 | A | -2.00 | — | 8.51 | A / 0 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +2.00 | — | 9.1397 | A / 0 |

- **Verbatim greedy continuations under fear@L22-24 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -45.69 | 5.3731 | — | 5 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +45.69 | 5.489 | — | 7 |
| trust.person.reciprocal.none.1 | A | -45.69 | 4.5177 | — | 0 |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | 4.9822 | — | 10 |
| trust.person.reciprocal.none.1 | P | -45.69 | — | 13.6152 | B / $15 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +45.69 | — | 18.0551 | E / 30 |
| trust.person.reciprocal.none.1 | A | -45.69 | — | 7.8245 | A / $5 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | — | 17.8605 | E / 30 |

- **Verbatim greedy continuations under joy@L22-24 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -45.69 | 5.4418 | — | 7 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +45.69 | 5.3067 | — | 5 |
| trust.person.reciprocal.none.1 | A | -45.69 | 5.0066 | — | 7 |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | 5.2826 | — | 7 |
| trust.person.reciprocal.none.1 | P | -45.69 | — | 14.6735 | B / 9 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +45.69 | — | 16.2276 | C / $15 |
| trust.person.reciprocal.none.1 | A | -45.69 | — | 15.8522 | B / 15 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | — | 13.8581 | C / 15 |

- **Verbatim greedy continuations under joy@L25 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -2.00 | 5.6783 | — | 5 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +2.00 | 5.7719 | — | 7 |
| trust.person.reciprocal.none.1 | A | -2.00 | 3.8686 | — | 0 |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +2.00 | 4.5326 | — | 7 |
| trust.person.reciprocal.none.1 | P | -2.00 | — | 12.2094 | B / 15 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +2.00 | — | 12.8634 | B / $9 |
| trust.person.reciprocal.none.1 | A | -2.00 | — | 9.9855 | A / 0 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +2.00 | — | 9.8048 | A / $0 |

- **Verbatim greedy continuations under sadness@L22-24 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -45.69 | 5.9807 | — | 7 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +45.69 | 5.015 | — | 7 |
| trust.person.reciprocal.none.1 | A | -45.69 | 5.7188 | — | 10 |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | 5.4687 | — | 10 |
| trust.person.reciprocal.none.1 | P | -45.69 | — | 13.7345 | B / 11 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +45.69 | — | 23.4467 | E / 30 |
| trust.person.reciprocal.none.1 | A | -45.69 | — | 13.2921 | A / 0 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | — | 16.6095 | C / $12 |

- **Verbatim greedy continuations under surprise@L22-24 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -45.69 | 4.6513 | — | 7 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +45.69 | 5.796 | — | 5 |
| trust.person.reciprocal.none.1 | A | -45.69 | 5.5326 | — | 10 |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | 5.9694 | — | 7 |
| trust.person.reciprocal.none.1 | P | -45.69 | — | 16.5655 | E / 30 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +45.69 | — | 18.2891 | E / 30 |
| trust.person.reciprocal.none.1 | A | -45.69 | — | 13.4049 | C / 15 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | — | 19.7507 | E / 30 |

- **Verbatim greedy continuations under valence@L22-24 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -45.69 | 5.5036 | — | 7 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +45.69 | 5.7829 | — | 7 |
| trust.person.reciprocal.none.1 | A | -45.69 | 5.8445 | — | 7 |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | 6.2365 | — | 7 |
| trust.person.reciprocal.none.1 | P | -45.69 | — | 15.9941 | B / $9 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +45.69 | — | 12.8913 | B / 9 |
| trust.person.reciprocal.none.1 | A | -45.69 | — | 11.4424 | A / $5 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | — | 14.8104 | C / 15 |

- **Verbatim greedy continuations under valence@L22-24|gen steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -45.69 | 5.3711 | — | 5 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.5331 | — | 7 |
| trust.person.reciprocal.none.1 | P | +45.69 | 5.4382 | — | 5 |
| trust.person.reciprocal.none.1 | A | -45.69 | 3.936 | — | 0 |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.506 | — | 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | 3.1011 | — | 0 |
| trust.person.reciprocal.none.1 | P | -45.69 | — | 12.2211 | B / $15 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +45.69 | — | 13.2697 | B / $9 |
| trust.person.reciprocal.none.1 | A | -45.69 | — | 9.7083 | A / 0 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +45.69 | — | 10.0177 | A / 0 |

- **Verbatim greedy continuations under valence@L32-34 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -47.54 | 4.3132 | — | I give you $10. |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +47.54 | 4.2105 | — | 0 |
| trust.person.reciprocal.none.1 | A | -47.54 | 0.3303 | — | 0 |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +47.54 | 4.8632 | — | 10 |
| trust.person.reciprocal.none.1 | P | -47.54 | — | 13.2495 | B / 6 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +47.54 | — | 16.2073 | E / 30 |
| trust.person.reciprocal.none.1 | A | -47.54 | — | 5.4236 | A / 0 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +47.54 | — | 14.8366 | B / 11 |

- **Verbatim greedy continuations under valence@L46-48 steering** (seed 0; extremes of the available doses):

| item | arm | α | amount EV | exp. return | text |
|---|---|---|---|---|---|
| trust.person.reciprocal.none.1 | P | -64.86 | 5.4784 | — | 7 |
| trust.person.reciprocal.none.1 | P | +0.00 | 5.539 | — | 7 |
| trust.person.reciprocal.none.1 | P | +64.86 | 6.8283 | — | 7 |
| trust.person.reciprocal.none.1 | A | -64.86 | 5.3192 | — | I'll go with 10! |
| trust.person.reciprocal.none.1 | A | +0.00 | 4.5743 | — | 0 |
| trust.person.reciprocal.none.1 | A | +64.86 | 7.2936 | — | 8 |
| trust.person.reciprocal.none.1 | P | -64.86 | — | 8.6171 | B / 12 |
| trust.person.reciprocal.none.1 | P | +0.00 | — | 12.6269 | B / $9 |
| trust.person.reciprocal.none.1 | P | +64.86 | — | 11.303 | B / 15 |
| trust.person.reciprocal.none.1 | A | -64.86 | — | 7.8386 | A / $15 |
| trust.person.reciprocal.none.1 | A | +0.00 | — | 8.7122 | A / 0 |
| trust.person.reciprocal.none.1 | A | +64.86 | — | 11.8421 | B / 15 |

