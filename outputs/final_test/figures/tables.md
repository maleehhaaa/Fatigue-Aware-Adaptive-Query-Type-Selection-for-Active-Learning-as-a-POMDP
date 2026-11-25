# Final-test tables

Made by scripts/make_final_test_figures.py from outputs/final_test.

## Table 1. Stage 2, the 12 main comparisons (QMDP minus rival, average accuracy 0–2,400 s, points)

| Setting | Starting labels | Rival | Difference | 95% CI | p | p (Holm, 12 tests) | Result |
|---|---|---|---|---|---|---|---|
| No fatigue | 50 | Ignores fatigue | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | not clear |
| No fatigue | 200 | Ignores fatigue | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | not clear |
| Markov, β = 1 | 50 | Greedy + rest | +0.03 | [-0.49, +0.60] | 0.9044 | 1.0000 | not clear |
| Markov, β = 1 | 200 | Greedy + rest | -0.27 | [-0.55, -0.02] | 0.0606 | 0.4244 | not clear |
| Markov, β = 2 | 50 | Greedy + rest | +0.07 | [-0.48, +0.66] | 0.8067 | 1.0000 | not clear |
| Markov, β = 2 | 200 | Greedy + rest | -0.46 | [-0.71, -0.20] | 0.0033 | 0.0300 | clearly worse |
| FALCON, β = 1 | 50 | Greedy | +0.06 | [-0.35, +0.50] | 0.7863 | 1.0000 | not clear |
| FALCON, β = 1 | 200 | Greedy | +0.28 | [+0.01, +0.54] | 0.0506 | 0.4046 | not clear |
| Lapse, β = 1 (onset in answers) | 50 | Ignores fatigue | -4.34 | [-5.69, -3.02] | 0.0000 | 0.0002 | clearly worse |
| Lapse, β = 1 (onset in answers) | 200 | Ignores fatigue | -4.58 | [-5.41, -3.73] | 0.0000 | 0.0000 | clearly worse |
| Lapse, β = 1 (onset in seconds) | 50 | Always 2-choice | -1.05 | [-2.94, +0.92] | 0.3194 | 1.0000 | not clear |
| Lapse, β = 1 (onset in seconds) | 200 | Always 2-choice | -1.84 | [-2.71, -0.93] | 0.0011 | 0.0110 | clearly worse |

## Table 2. Hypotheses on the final test

| Hypothesis | Result | Evidence |
|---|---|---|
| H1: no fatigue, QMDP within 1 SE of the best required baseline | holds | difference 0.00 (SE 0.00) against Ignores fatigue |
| H2: beats all of B1–B8 at β ≥ 1 (Holm) | Markov holds; FALCON partial; Lapse fails | Markov 16/17, FALCON 1/3, core Lapse 0/4, Lapse (seconds) 0/4 |
| H3: lead over greedy grows with β (Markov) | fails | gaps 50.0, 45.4, 39.8, 43.4; slope -2.70 per β [-4.50, -1.02] |
| Robustness: beats greedy under misspecified fatigue (Holm) | 5/6 | fails in the FALCON world |
| Stage 2: better than the named rival (Holm, 12 tests) | not shown | 0 clearly better, 4 clearly worse |

## Table 3. Stage 2 means, 200 starting labels

| Setting | Policy | Average accuracy | Accuracy at 2,400 s | Answers | Wrong labels |
|---|---|---|---|---|---|
| No fatigue | QMDP | 78.68 | 79.62 | 3515 | 1.6% |
| No fatigue | Greedy + rest | 78.68 | 79.62 | 3515 | 1.6% |
| No fatigue | Greedy | 78.68 | 79.62 | 3515 | 1.6% |
| No fatigue | Ignores fatigue | 78.68 | 79.62 | 3515 | 1.6% |
| No fatigue | Always 2-choice | 78.74 | 80.46 | 2377 | 2.3% |
| No fatigue | Always verify | 78.26 | 78.71 | 3742 | 1.3% |
| Markov, β = 1 | QMDP | 77.34 | 79.38 | 2157 | 3.6% |
| Markov, β = 1 | Greedy + rest | 77.61 | 79.86 | 2187 | 3.2% |
| Markov, β = 1 | Greedy | 76.70 | 79.29 | 1671 | 5.5% |
| Markov, β = 1 | Ignores fatigue | 76.11 | 79.11 | 1199 | 7.1% |
| Markov, β = 1 | Always 2-choice | 76.61 | 79.48 | 1043 | 5.5% |
| Markov, β = 1 | Always verify | 76.46 | 78.89 | 1843 | 4.7% |
| Markov, β = 2 | QMDP | 77.31 | 79.52 | 2179 | 4.2% |
| Markov, β = 2 | Greedy + rest | 77.76 | 80.18 | 2269 | 2.5% |
| Markov, β = 2 | Greedy | 76.48 | 78.95 | 1662 | 6.5% |
| Markov, β = 2 | Ignores fatigue | 75.67 | 78.73 | 1188 | 11.6% |
| Markov, β = 2 | Always 2-choice | 76.06 | 78.65 | 1024 | 10.5% |
| Markov, β = 2 | Always verify | 76.32 | 78.60 | 1805 | 6.0% |
| FALCON, β = 1 | QMDP | 77.20 | 79.39 | 2041 | 4.1% |
| FALCON, β = 1 | Greedy + rest | 76.72 | 79.03 | 1729 | 4.6% |
| FALCON, β = 1 | Greedy | 76.92 | 79.38 | 1884 | 4.7% |
| FALCON, β = 1 | Ignores fatigue | 76.53 | 79.68 | 1389 | 6.0% |
| FALCON, β = 1 | Always 2-choice | 76.97 | 79.86 | 1130 | 5.1% |
| FALCON, β = 1 | Always verify | 77.04 | 79.10 | 2064 | 4.3% |
| Lapse, β = 1 (onset in answers) | QMDP | 70.24 | 66.75 | 2054 | 30.9% |
| Lapse, β = 1 (onset in answers) | Greedy + rest | 71.76 | 68.17 | 2008 | 31.4% |
| Lapse, β = 1 (onset in answers) | Greedy | 73.20 | 72.20 | 1535 | 25.6% |
| Lapse, β = 1 (onset in answers) | Ignores fatigue | 74.82 | 75.54 | 1110 | 22.9% |
| Lapse, β = 1 (onset in answers) | Always 2-choice | 74.79 | 75.91 | 1000 | 19.7% |
| Lapse, β = 1 (onset in answers) | Always verify | 71.46 | 69.41 | 1714 | 27.1% |
| Lapse, β = 1 (onset in seconds) | QMDP | 70.37 | 67.98 | 2033 | 29.7% |
| Lapse, β = 1 (onset in seconds) | Greedy + rest | 70.43 | 67.86 | 1898 | 33.2% |
| Lapse, β = 1 (onset in seconds) | Greedy | 72.06 | 71.06 | 1506 | 29.3% |
| Lapse, β = 1 (onset in seconds) | Ignores fatigue | 71.82 | 71.70 | 917 | 35.2% |
| Lapse, β = 1 (onset in seconds) | Always 2-choice | 72.21 | 71.79 | 966 | 33.0% |
| Lapse, β = 1 (onset in seconds) | Always verify | 71.77 | 70.56 | 1705 | 26.0% |
