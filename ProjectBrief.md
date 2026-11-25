# Project Brief (Revised): Fatigue-Aware Adaptive Query-Type Controller

- **Course:** Advanced Machine Learning (MLR570), American University of Sharjah
- **Team:** Maleeha Fatima, Sara Albanna Alraeesi, Shaikha Alzaabi
- **Status (23 November 2026):** implementation, experiments and the final test are complete. `src/` is frozen (fingerprint `9d5b9588c1550a431a1f1c42075e71bebae04210cca85ab7bc2ef9e50d3ae143`). Only the report remains.
- **This revision:** replaces the 13 November brief. It describes the project **as implemented** in the frozen code and the executed notebooks 00–11. Design decisions, the simulator specification and the hypotheses are kept. Prototype tables, the 13 November checkpoints, the ten-day plan and the prototype code are removed because the final runs supersede them. Appendix A lists every point where the original brief differs from the implementation.

---

## 1. The project in brief

### 1.1 Idea

A classifier is trained with active learning. When the model is unsure about an image, it can ask a human, in one of three formats that differ in difficulty for the human:

| Format | Example | What the learner gets from the answer |
|---|---|---|
| **Verify** (easiest) | "Is this a cat?" | yes → the class; no → "not cat" |
| **Two-choice** | "Cat, dog, or neither?" | a pick → the class; neither → "not cat, not dog" |
| **Full label** (hardest) | "Which of the 10 classes?" | the class |

Over a session the human tires, and tired people answer more slowly and make more mistakes. A **controller** tracks the human's hidden fatigue from response times and, at every turn, decides whether to **pause** (let the person rest) or which format to ask. Because a hard question now makes later answers worse, the problem is a **partially observable Markov decision process (POMDP)**. The objective is the most information about the true labels for the fewest human seconds.

### 1.2 Research question and objectives

**Research question.** Can a fatigue-aware POMDP controller, which decides both *whether* to ask and *which question type* to ask, obtain more information and higher model accuracy per second of human effort than fixed question types and existing adaptive querying rules?

| Sub-question | Tested by |
|---|---|
| **RQ1.** Does adapting the format to estimated fatigue beat every fixed format (verify, two-choice, full, random)? | H2; Stage 1 and Stage 2 comparisons |
| **RQ2.** Does planning ahead add value beyond a fatigue-aware but greedy controller? | H3; QMDP vs greedy, greedy plus rest, E3 ablation |
| **RQ3.** How does the benefit depend on the fatigue model, its strength β, its distribution across people, recovery, and the zero-fatigue setting? | 28-case initial study, 7,812-condition grid |
| **RQ4.** How robust is the controller when its fatigue model is wrong? | Robustness cases; deployable QMDP; FALCON and Lapse worlds |

**Objectives:** (1) build a simulated annotator calibrated to CIFAR-10H, with fatigue as an explicit, varied assumption; (2) implement and compare three controllers (QMDP, belief-grid POMDP, Double DQN) against eight required baselines and further rivals; (3) measure the effect on a real classifier retrained from the simulated answers (Stage 2); (4) select the main controller and evaluate it once on held-out people and images.

### 1.3 Answer in one paragraph (final test)

Under Markov (recoverable) fatigue, QMDP beats all eight required baselines on information per human second (16 of 17 Markov cases after Holm correction). It does no harm without fatigue (H1). Its lead over greedy does **not** grow with fatigue strength (H3 fails). It is fragile when its fatigue model is wrong: it beats the baselines in only 1 of 3 FALCON cases at β ≥ 1 and in 0 of 8 Lapse cases. When the classifier is retrained from the answers (Stage 2), QMDP is never clearly better than the rival named in advance (0 of 12) and is clearly worse in 4 of 12: the three Lapse cells, and Markov β = 2 with 200 labels against a tuned greedy-plus-rest rule. Pausing is the main benefit, and a simple rest rule captures nearly all of it. Section 10 gives the numbers.

---

## 2. Decisions

### 2.1 Original decisions (13 November) and their outcome

| # | Decision | Reason | Outcome |
|---|---|---|---|
| D1 | Dataset: **CIFAR-10H** and the CIFAR-10 test images | The only large dataset with annotator ID, trial order and per-trial reaction time | Kept. Other datasets not used (§15.2) |
| D2 | Actions: **verify / two-choice (with "neither") / full label**, plus **don't ask** | The user's definition | Kept. "Don't ask" is implemented as a **10-s pause** (D11). A gold check is a fifth action for the grid planner and DQN |
| D3 | Main controller: **best on validation** among QMDP, belief-grid POMDP and DQN, by a rule fixed in advance | User instruction | Applied on the full 7,812-condition grid: **QMDP** (§7.6) |
| D4 | The human is **simulated**, calibrated to CIFAR-10H and literature values | No dataset records long sessions with several formats and a fatigue measure | Kept |
| D5 | Effort is measured in **seconds**, not questions | Formats differ in cost | Kept |
| D6 | Reward = **mutual information − λ × seconds** | Per-step accuracy change is too noisy | Kept; λ = 1.0 (D10) |
| D7 | Features: **frozen ResNet-18 (ImageNet)** + linear head | Already built; CLIP would make CIFAR-10 too easy | Kept |
| D8 | The **zero-fatigue setting is always reported** | CIFAR-10H shows no average fatigue | Kept (H1) |
| D9 | Professor's comparisons are mandatory: all-random, all-easy, all-hard, several fatigue distributions | Supervisor request | Kept (B1–B4; four β distributions) |

### 2.2 Decisions made during the project

| # | Decision | Reason | Source |
|---|---|---|---|
| D10 | **λ = 1.0 bits per second** (was 0.25) | Chosen on training people by a rule fixed before the sweep; close to the theory value 0.958 (median Dinkelbach fixed point of QMDP's bits per second); fair to the λ-aware baselines | - |
| D11 | **Pause = 10 s**, recovery probability 0.35 per pause | The old 0.16-s skip made rest nearly free | Review finding 2 |
| D12 | **fatigue_scale = 0.1** on all fatigue increments | The old rates exhausted a person in about 15 full labels; real people give 200 labels in about 9 min with flat accuracy | Finding 1 |
| D13 | **FALCON fatigue measured from the start of the decline**, starts at 0, capped at 4; peak quality U(0.8, 0.99) | The old version counted warm-up as fatigue and exploded as the peak approached 1 | Finding 4 |
| D14 | **Forced lapse**: one lapse per session, onset counted in answers (`forced`); a second world counts it in seconds (`forced_seconds`, added 21 Nov) | The calibrated rare hazard produced almost no lapses, so Lapse duplicated Markov | Finding 5 |
| D15 | New rivals: **greedy plus rest** (rest threshold tuned on training people) and **deployable QMDP** (one fixed model in every world) | Greedy almost never pauses, and the main QMDP is given the true world's population parameters | Findings 6, 10 |
| D16 | **SpeedTracker**: the fatigue filter learns each person's speed from their first 20 answers | A naturally slow person looked permanently tired | Finding 7 |
| D17 | **Code freeze** before the full grid and the final test; the final test is run once, with settings written beforehand | Reproducibility; no tuning on test data | - |

The pre-fix simulator is kept as two labelled **fast-stress** Stage-1 cases, so the earlier results stay reproducible.

---

## 3. Background and motivation

### 3.1 Lessons from the first pipeline (4 November update)

The first pipeline (frozen ResNet-18; 200-image seed; a simulated annotator with an AR(1) load L_{t+1} = ρL_t + c(a_t); complementary labels for verification; threshold controllers) showed:

1. **Verification-only always won.** The controller beat an alternating mix at β = 0.2 (+1.23 pp, CI +0.57 to +1.89) but tied verify-only (−0.01 pp, CI −0.35 to +0.33).
2. In that simulator verify was cheapest, caused the least load and was nearly as reliable as a full label. When one action is best in every state, no adaptive controller can beat it.
3. Fatigue reduced every format equally, so fatigue awareness never changed the best format.
4. The fitted load sensitivity was tiny (β ≈ 0.0041, bootstrap −0.0002 to +0.0075).
5. A per-step validation-accuracy reward was pure noise.
6. The controllers knew the simulator's parameters (a validity problem).

**Design principle.** Adaptation can only pay off if the best format depends on the hidden state. The simulator therefore has literature-grounded asymmetries: fatigue degrades demanding formats more than easy ones; demanding formats and hard images tire people faster; tired verifiers develop a "yes" bias; not asking lets the person recover. FALCON (the closest prior work) names instance-dependent fatigue as future work; the format- and difficulty-dependent fatigue used here is that extension.

### 3.2 CIFAR-10H facts (whole dataset, own analysis of `cifar10h-raw.csv`, 13 Nov)

Source: <https://github.com/jcpeterson/cifar-10h>. 539,910 records from 2,571 people, 210 trials each: 514,200 ordinary trials and 25,710 attention checks (positions 20, 41, …, 209; 98.2% accuracy). Image IDs follow the original CIFAR-10 test order.

- **Accuracy** 94.9%; about 51 labels per image (50–63).
- **Image difficulty is skewed:** 35.6% of images are answered correctly by everyone; 1.2% are below 50%. Shrunk estimate p̂_x = (c + 0.5)/(n + 1), d_x = logit(p̂_x) (higher = easier); quantiles of d_x: 1% −0.12, 5% 1.15, 25% 2.97, 50% 3.52.
- **Annotator skill** θ_j (logit residual after difficulty): SD 0.62, skew −1.47 (long tail of weak annotators), so real people are resampled rather than drawn from a normal distribution. Errors per person (of 200): mean 10.3, variance 131 (over-dispersed).
- **Top confusions:** deer→horse, cat→dog, dog→cat, truck↔automobile, deer→dog, airplane→ship.
- **Reaction time:** median 1,483 ms; log-RT SD 0.46, right-skewed; 12 negative values, 114 over 60 s, maximum about 2 h (clipped to 0.2–30 s). Errors are slower (median 2.21 s vs 1.47 s). Median session 8.9 min (IQR 7.5–10.9).
- **No average fatigue:** the difficulty-adjusted residual stays within ±0.2 pp across 20-trial bins. RT falls from 1.59 s to 1.42 s, which reflects practice. The survey's own analysis gives −0.12 pp per 100 trials in CIFAR-10H (CI −0.25 to +0.01) and +2.03 in ImageNet-16H (learning). Hence the fatigue effect is **swept, including zero**.
- **Individual drift is mostly disengagement:** per-person accuracy-slope SD 3.58 pp/100 trials vs 2.3 under a shuffled null; for people above 90% accuracy, drift equals the null. The excess comes from about 16 people (0.6%) who collapse (example: annotator 2342 falls 96% → 58% → 42% → 10% across quarters).

The development subset used in the notebooks (training people, non-final-test images) gives slightly different values: 246,895 answers, 94.91% accuracy, median 1.49 s.

---

## 4. Problem formulation

Each episode is one session. In Stage 1 it has **T = 150 steps** (300 in long-session cases). In Stage 2, sessions are 600-s shifts within a seconds budget. At each step one candidate image x_t arrives from the active-learning stream.

**State.**
- Hidden: fatigue F_t ∈ {0, 1, 2, 3, 4} (0 = fresh). In Lapse worlds the person also has an engaged/disengaged flag, which **no controller models**.
- Observed: the classifier's predictive distribution p(·|x_t) (top-1/top-2 classes and probabilities), the predicted human easiness d̂(x_t), steps (or seconds) remaining, and the controller's own action history.
- Belief: b_t(F) = P(F_t = F | history), from a forward filter (§7.1). Every session starts with b = (1, 0, 0, 0, 0).

**Actions** (ŷ₁, ŷ₂ = the model's top-two classes):

| a | Action | Human time | Label information |
|---|---|---|---|
| 0 | Pause (don't ask) | 10 s fixed | none; fatigue may fall one level (probability ≈ 0.35) |
| 1 | Verify "Is this ŷ₁?" | 0.35 × the person's RT draw | yes → ŷ₁; no → "not ŷ₁" |
| 2 | Two-choice "ŷ₁, ŷ₂ or neither?" | 0.6 × RT draw | pick → class; neither → "not ŷ₁, not ŷ₂" |
| 3 | Full label (10-way) | 1.0 × RT draw | class |
| 4 | Gold check (a known-answer verify item) | 0.35 × RT draw | none; its correctness is observed |

QMDP, greedy, greedy plus rest and fatigue-blind choose among actions 0–3; QMDP never pays for information, so gold is excluded by design. The grid planner and the DQN may choose action 4. Stage 2 runs only policies that use actions 0–3.

**Transitions.** Fatigue rises after questions (more for demanding formats and hard images) and may fall after a pause (§5.4).

**Observations.** After a question: its response time; after a gold check, also whether it was correct. The correctness of an ordinary answer is **never observed**.

**Reward.** r_t = I(Y; answer | p, F) − λ · seconds(a_t), with λ = 1.0 bits per second.
- I(Y; answer) is the mutual information between the true class and the answer, with the classifier's prediction as the prior over Y and the person's answer channel at fatigue F (§5.3). Pauses and gold checks give zero information; a lapsed person's random answers also give zero.
- Noisy (tired) answers carry less information, so no separate error penalty is needed.
- λ is the price of a human second (FALCON's Lagrangian budget view; original §5.5). It was chosen by the λ sweep (D10); the sweep and an information–time frontier (λ = 0.05–1) are reported as sensitivity results. At λ = 1 nearly every policy has a negative net reward, as expected when λ is close to the achievable bits per second.
- I(Y; answer) is only meaningful if p is calibrated, so **temperature scaling** is applied after every retrain (§6).

---

## 5. Simulated annotator (as implemented)

The simulator lives in `src/sim/` and owns all ground truth (true labels, F, θ_j, β_j, lapse state, person parameters). Controllers receive only an observation dictionary (answer, seconds, gold flag, gold correctness) plus public model outputs. Unit tests enforce this boundary.

### 5.1 People and calibration (`sim/calib.py`, Notebook 03)

- Each simulated person is a **resampled real CIFAR-10H participant** from the relevant split (train for tuning and DQN training, validation for development, test for the final test). People are split 60/20/20 (1,542 / 514 / 515), with a new seed-42 split because the original test IDs were not supplied.
- Fitted on development images only (final-test images are never used):
  - Difficulty d_x = logit((c + 0.5)/(n + 1)), from training people's answers.
  - Wrong answers: per-image human confusion h_x(·) from training people's errors, with a class-level confusion matrix as fallback.
  - Each person's own profile, for every split, from that person's development answers (held-out people can then be simulated, but they never shape the population values):
    - Skill θ_j: solved so that the mean of σ(d_x + θ_j) over the person's images matches their smoothed accuracy, plus one shared offset so that training people match overall accuracy.
    - Speed: μ_j, σ_j = mean and SD of the person's log-RT residual after removing 0.8[1 − σ(d_x)], 0.4·1[wrong] and −0.06·trial/100 (σ_j ≥ 0.08).
- Public population values (training people) that controllers may use: skill 0.524, log-speed μ 0.484, σ_pop = √(mean σ_j² + var μ_j) = 0.447, SD of personal speed 0.228, within-person noise 0.384, and the class confusion matrix.

### 5.2 Image difficulty and the deployable predictor (`learn/difficulty.py`)

Controllers never see the true d_x of a pool image. A small MLP (512 → 32 → ReLU → 1; AdamW lr 0.01, weight decay 0.05, 300 epochs, MSE on d_x, output clipped to [−4, 6]) predicts d̂_x from the frozen features. It is trained only on the 2,000 difficulty-training images. On validation images its MSE is 1.099 against 1.116 for a constant, so **the controller effectively sees constant difficulty** (a limitation; a BCE option exists but was not adopted).

### 5.3 Answer model (`sim/channel.py`)

Probability that question type a about image x is answered correctly at fatigue F:

  **P(correct | a, x, F) = σ(d_x + θ_j + δ_a − β_j · w_a · F)**

| | Verify | Two-choice | Full | Gold |
|---|---|---|---|---|
| δ_a (ease bonus) | 1.5 | 1.0 | 0 | 1.5 (on an easy item, d = 4) |
| w_a (fatigue weight) | 0.25 | 0.5 | 1.0 | 0.25 |

δ_full = 0 reproduces CIFAR-10H at F = 0. Fatigue hurts demanding formats more.

Wrong-answer models:
- **Full label:** the wrong class is drawn from h_x(·) (class confusion if no one erred on x).
- **Verify ŷ₁:** if ŷ₁ = y, "yes" with P(correct); if ŷ₁ ≠ y, "yes" with clip(1 − P(correct) + 0.04·F, 0, 1) (**yes-bias under fatigue**; 0.04 per level, set to 0 when β = 0).
- **Two-choice:** if y ∈ {ŷ₁, ŷ₂}, the correct pick with P(correct), otherwise the other class or "neither", half each; if y ∉ {ŷ₁, ŷ₂}, "neither" with P(correct), otherwise ŷ₁ or ŷ₂ in proportion to their confusion weights.
- **Lapsed person:** every format is answered uniformly at random (gold: 50% correct).

Every channel row sums to one, and the information lies between 0 and the prior entropy (tested).

### 5.4 Fatigue worlds (`sim/fatigue.py`, `sim/world.py`)

All fatigue stories are hypotheses, not measured laws.

- **None** (β = 0): F stays 0; the CIFAR-10H-calibrated control.
- **Markov (main model):** after a question of type a on image x, the fatigue level rises by one with probability
  clip(u_a × 0.1 × s_j × [2 − σ(d_x)], 0, 1), with u = 0.06 (verify), 0.12 (two-choice), 0.25 (full), 0.06 (gold, d = 4).
  s_j ~ U(0.5, 1.5) × speed is a per-person rate multiplier, and 0.1 is `fatigue_scale` (D12). With 2 − σ(d_x) = 1 + (1 − p̂_x), this is the original rule u_a(1 + η(1 − p̂_x)) with η = 1, scaled by 0.1.
  - After a pause, fatigue falls one level with probability r_j = clip(r + U(−0.15, 0.15)), with r = 0.35 by default. F is capped to [0, 4].
  - Always-full reaches F ≈ 1.3 after 50 and 3.1 after 150 questions.
  - **Binomial interpretation** (professor's request): each question is a Bernoulli trial for "fatigue goes up one level", so k questions of one type give Binomial(k, p) increments before capping and recovery.
- **FALCON:** fatigue follows cumulative workload W, which increases by c_a·[2 − σ(d_x)] per question (c = 0.25 / 0.5 / 1, gold 0.25). Quality follows FALCON's decline q(W) = q_base + (q_peak − q_base)·σ(−k(W − ρ̄L)), with L = session length.
  - Fatigue is **F = clip(logit q_start − logit q(W), 0, 4)**, where q_start is the decline curve's quality at W = 0. F starts at 0 and never falls, because the default has no recovery.
  - FALCON's warm-up phase is treated as practice, which the RT model already contains (`falcon_quality` keeps the full warm-up curve for reference only).
  - Per-person parameters come from FALCON's CIFAR-100 ranges (Table 2): q_base U(0.4, 0.5), q_peak U(0.8, 0.99) (capped below 1), ρ̄ U(0.25, 0.5), k U(0.05, 0.1).
  - The planner uses a five-state chain with up-probability reach_k × load / gap_k, from 20,000 sampled curves. It expects F = 2.37 after 150 full labels, matching the world.
- **Lapse:** Markov fatigue plus **one forced lapse per session**. After the lapse the person answers at random until the session ends; there is no recovery, and the lapse does not depend on fatigue.
  - `forced`: the onset comes after U(20, 100) answered questions in Stage 1, or U(100, 300) per 600-s shift in Stage 2 (gold checks count).
  - `forced_seconds`: the onset comes after U(14, 88) s in Stage 1, or U(75, 310) s per shift in Stage 2, with pauses counted.
  - The ranges are set to where always-verify meets the lapse in the answers world, so a λ-free reference meets about the same lapse in both worlds. The onset draw uses its own random stream and is the same for every policy.
  - The original σ(α₀ + α₁F) chain, calibrated to 0.6% of people within 200 questions, survives only as the legacy `rare` mode: in the calibration replay and the fast-stress cases.
- **Fast-stress cases:** the pre-review setting: fatigue_scale 1.0, a skip of 0.1 × exp(μ_pop) ≈ 0.16 s, and `rare` lapses. It is kept as two labelled Stage-1 cases (Markov β = 1, 2).

### 5.5 Fatigue sensitivity across people

β_j has mean β̄ ∈ {0, 0.5, 1, 2, 3} (logit units per fatigue level). β controls how much a given level hurts answers, not how fast fatigue builds.

| Distribution | β_j |
|---|---|
| Constant | β̄ |
| Uniform | U(0, 2β̄) |
| Bernoulli mixture ("susceptible vs resistant") | β̄/0.3 with probability 0.3, else 0 |
| Gamma (heavy tail) | Gamma(shape 2, scale β̄/2) |

Per-person randomisation, as in FALCON's domain randomisation: rate multiplier s_j ~ U(0.5, 1.5) and recovery r ± 0.15.

### 5.6 Response time (observation model)

  ln τ = μ_j + 0.8[1 − σ(d_x)] + ln c_a + η · σ_pop · F + 0.4 · 1[wrong] − 0.06 · q/100 + σ_j ε,  ε ~ N(0, 1)

- c_a is the relative format time (§5.7); a gold check uses d = 4.
- 0.4 ≈ ln(2.21/1.47), the CIFAR-10H gap between wrong and correct answers.
- q is practice: the opportunity index in Stage 1; answered questions capped at 200 per person in Stage 2.
- η is the **signal** (0.5 / 1 / 2). At η = 1 and F = 4 the fatigue term multiplies time by exp(4 × 0.447) ≈ 6.0, so fatigue is both costly and easy to detect.
- The signal axis changes how much fatigue slows answers and how well it can be detected at the same time; it is not a pure noise experiment.

### 5.7 Time costs

Relative to a full label (1.0): verify 0.35 (Beck et al.: verifying is 3–4× faster than correcting), two-choice 0.6, gold 0.35. These multiply the person's own RT draw, which is about 1.5 s for a full label at F = 0. A pause is a fixed **10 s** (D11). The sensitivity grids vary the verify ratio over {0.25, 0.35, 0.5} and the two-choice ratio over {0.5, 0.6, 0.75}.

### 5.8 Simulator validation gate (Notebook 03)

With no fatigue and full labels only, the simulator replays the training sessions with newly sampled answers. It must pass eight coarse tolerances before any controller work, and it passes all eight:
- accuracy within 1 pp (0.949 vs 0.946)
- median time within 15%
- log-time SD within 15%
- wrong/correct time ratio within 20%
- mean drift slope within 0.5 pp/100
- drift SD within 25%
- confusion rank correlation ≥ 0.8
- legacy lapse onset within 0.15 pp of 0.6% per 200 questions

These are internal-consistency checks. Two are circular (the lapse onset, and the replay uses the same people), and two passing checks overshoot (drift SD 4.50 vs 3.74; time ratio 1.66 vs 1.50). They do not validate the fatigue assumptions.

---

## 6. Learning from each answer (Stage 2, `learn/`)

- **Classifier:** frozen ResNet-18 features (512-d, L2-normalised) and a linear head, trained with AdamW (weight decay 0.01). The initial fit on the trusted starting labels runs 200 epochs at lr 0.05. Every **10 acquired answers** the same head and optimiser continue for 10 epochs at lr 0.01 on all accumulated targets (warm start). The temperature is then recalibrated: 81 values from 0.05 to 20, on 1,000 validation calibration images.
- **Targets as allowed class sets**, with one partial-label loss −ln Σ_{k∈S} p_θ(k|x). It equals cross-entropy for a full label and the complementary loss for "no" (S = 9 classes) and "neither" (S = 8).
- **Reliability weights:** w = Σ_F b(F) · P̂(correct | a, x, F), from the public model and the belief, computed before the answer is seen. Trusted starting labels have weight 1. The loss is the weighted mean. This reduces the influence of doubtful answers; it is not a full noise-channel correction.
- The Bayesian soft-label option in the original brief was **not implemented**.

---

## 7. Controllers (`src/control/`)

### 7.1 Belief tracking (`belief.py`, `eval/stage1.py`)

- Residual: z = [ln τ − (μ_pop + 0.8(1 − σ(d̂)) + ln c_a − 0.06q/100) − o_j] / σ_pop, where o_j is the **SpeedTracker** offset. o_j is learned from the person's first 20 answers and shrunk toward 0 by (noise SD / prior SD)², using response times only.
- Likelihood (correctness is hidden, so it is a mixture of correct and slow wrong answers): L_F(z) = α_F φ(z − ηF) + (1 − α_F) φ(z − ηF − Δ), with Δ = 0.4/σ_pop and α_F the population-model correctness. A gold check uses only the observed branch. A pause gives no observation.
- **Update, then predict:** the observation updates the state that produced it, then the action's transition predicts the next state.

### 7.2 C1 — QMDP (`qmdp.py`, main controller)

1. Backward dynamic programming over F with the population model: V_t(F) = E_x[max_{a∈0..3} (r(F, a, x) + Σ_{F'} P(F'|F, a, x) V_{t+1}(F'))]. The expectation is the mean over contexts: all 2,000 Stage-1 contexts, or 64 representative candidates in Stage 2.
2. Act: a_t = argmax_a Σ_F b_t(F) [r(F, a, x_t) + Σ_{F'} P(F'|F, a, x_t) V_{t+1}(F')] on the actual image.
3. Stage 2: the tables are re-solved after every retrain. The receding horizon is min(150, remaining shift seconds / pause seconds).

QMDP assumes the state becomes known after one step, so it never pays for information (gold). The main QMDP uses the **true world's population-level model** (fatigue family, β̄, recovery, signal), but never the individual person's parameters.

### 7.3 Deployable QMDP (`qmdp_fixed`)

The same planner with one fixed assumed model in every world: Markov, β = 1, constant, r = 0.35, signal 1. It copies only the design choices (costs, horizon, λ, pause length). It shows what is achievable without knowing the true world.

### 7.4 C2 — Belief-grid POMDP (`pomdp_grid.py`)

- Belief simplex at resolution 0.1 (1,001 points).
- Observations: 5 response-time bins, and 10 outcomes for gold (time bin × correct or incorrect).
- Backward value iteration over 16 representative image contexts, with nearest-grid-point projection.
- Acts by a **one-step look-ahead with the actual image** plus the grid values (review fix; the old nearest-context lookup is kept as `stored_action`).
- Can choose gold checks, but in practice almost never does (≤ 0.07 per session).
- Known artefact: it over-pauses at recovery 0.5 (nearest-grid rounding).

### 7.5 C3 — Double DQN (`dqn.py`, `eval/train_dqn.py`)

- Input (16): belief (5), top-two probabilities, normalised entropy, d̂/4, remaining-session fraction, budget fraction (1 in Stage 1), last 5 actions. Network: 16 → 128 → 128 → 5.
- Training on training people only, 80 rounds × 32 parallel sessions × 150 steps (384,000 interactions). Each round draws a world: Markov (weight 3), FALCON, Lapse or None, with random β, distribution, signal and recovery.
- Optimisation: ε-greedy (1 → 0.05 over 75% of the rounds); replay buffer 60,000, batch 256; **4 updates per step (≈ 48,000)**, γ = 0.99, rewards × 0.1, soft target update 0.01, Adam 3 × 10⁻⁴, smooth-L1 loss, gradient clip 10.
- Used in Stage 1 only (the planned Stage-2 fine-tuning was not done).
- In the Markov validation sessions it chooses about 24 gold checks per 150-step session (time with no label information) and about 1 pause.

### 7.6 Main-controller selection (rule fixed in advance, original §8.4)

The highest mean Stage-1 net reward on **validation people**, over all fatigue conditions except None, averaged within each matched session first. Within 1 SE of the best, the simpler controller wins (QMDP > grid > DQN).
- Initial study (28 cases, core set): QMDP −45.5, DQN −67.7, grid −77.0.
- **Full grid (7,812 conditions, 15 policies, 500 sessions): QMDP −88.96 (SE 2.98), DQN −202.02 (3.59), grid −222.76 (3.76).** QMDP is selected by about 38 SE, so the tie rule is not needed.

---

## 8. Baselines and comparison policies (`control/baselines.py`, `qmdp.py`)

| # | Policy (code name) | Rule as implemented | Stage 1 | Stage 2 |
|---|---|---|---|---|
| B1 | Always full (`full`) | Action 3 | ✓ | ✓ |
| B2 | Always two-choice (`two_choice`) | Action 2 | ✓ | ✓ |
| B3 | Always verify (`verify`) | Action 1 | ✓ | ✓ |
| B4 | Random format (`random`) | Uniform over 1–3 | ✓ | ✓ |
| B5 | Alternating (`alternating`) | 1 + t mod 3 | ✓ | – |
| B6 | Exponential decay (`exp_decay`), Banerjee et al. 2025 | Ask with probability e^(−0.01t), random format; otherwise pause | ✓ | – |
| B7 | Gap and workload (`gap_workload`), Banerjee-style | u = 1 − (p₁ − p₂); full if u > 0.25·E_b[F], else verify if u > 0.1, else pause | ✓ | – |
| B8 | Fatigue-blind information per time (`blind`), Martín-Urcelay-style | argmax_a r(F = 0, a, x) | ✓ | ✓ |
| B9 | Greedy fatigue-aware (`myopic`) | argmax_a Σ_F b(F) r(F, a, x) | ✓ | ✓ |
| B9+ | Greedy plus rest (`rest_threshold`) | Greedy, but pause when E_b[F] ≥ threshold; threshold ∈ {1, 1.5, 2, 2.5, 3} tuned on 200 training sessions per case (Stage 2: per setting, same procedure) | ✓ | ✓ |
| – | Deployable QMDP (`qmdp_fixed`) | §7.3 | ✓ | ✓ |
| C2, C3 | Grid planner (`grid`), DQN (`dqn`) | §7.4–7.5 | ✓ | – |
| B10 | Oracle (`oracle`) | QMDP tables at the true (rounded) fatigue; a reference, not an upper bound | ✓ | – |
| B11 | Random image + full label (`random_image_full`) | Random remaining pool image, action 3 | – | ✓ |

B1–B8 are the **required baselines** for H2. Greedy, greedy plus rest and the deployable QMDP are additional rivals, not part of H2. B12 (the old threshold schedule) was optional and not implemented. The B6/B7 rules are transparent adaptations, not full reproductions of the published methods.

---

## 9. Evaluation protocol

### 9.1 Data roles

| Data | Training | Validation | Test |
|---|---|---|---|
| People (60/20/20) | Calibration of population values, rest-threshold tuning, λ sweep, DQN training, E1b | All development studies | Final test only |
| Images per class | Pool 400 (active-learning candidates), difficulty-training 200 | 200, split per class into 100 calibration + 100 evaluation | 200 (final test only; features extracted only for the final test) |

Classifier temperature is always calibrated on the 1,000 validation calibration images. Development accuracy is scored on the 1,000 evaluation images; the final test scores all 2,000 final-test images.

### 9.2 Stage 1 — information level (no retraining)

- **Contexts:** the four Notebook-02 classifier checkpoints (50 / 200 / 500 / 1,000 clean labels) each give their 500 most uncertain remaining pool images (2,000 contexts with calibrated probabilities and d̂). Each step draws one context at random.
- 500 matched sessions per policy and case. People, image streams and random tapes are shared across policies (common random numbers).
- Recorded per session: net reward, bits, seconds, bits per second, error rate, action counts, tracking error.

### 9.3 Stage 2 — real classifier in the loop

- 50 or 200 trusted starting labels (identical within a pair; their cost is excluded).
- Each turn draws a random image from the current **top-500 most uncertain** remaining pool images, refreshed after each retrain. The policy chooses the format or a pause.
- Human time runs in **600-s shifts**: a new person each shift, with fatigue reset and the belief and SpeedTracker restarted. A response crossing a shift boundary counts its partial time, but its answer is discarded (censored). Practice is capped at 200 answers per person.
- Each run continues to 2,400 s and is scored at 300 / 600 / 1,200 / 2,400 s, using the last model trained by then.
- **Primary metric:** average accuracy over 0–2,400 s (area under the staircase accuracy–time curve, divided by 2,400; "AUC"). Also: accuracy at each budget, acquired answers and their error rate, and time to 80% (primary target) and 75% (secondary target, declared before the study).
- Development: 5 settings × 10 policies × 2 sizes × 10 seeds (42–51) = **1,000 runs**. The settings are no fatigue, Markov β = 1 and 2, FALCON β = 1, and Lapse β = 1 (answers).

### 9.4 Experiment grids

| Set | Content | Use |
|---|---|---|
| Initial study (`initial_cases`, 28) | None; Markov, FALCON and Lapse at β 0.5 / 1 / 2 / 3; at Markov β = 2: 3 distributions, recovery 0.2 / 0.5, signal 0.5 / 2, 4 cost pairs, 300 steps; Lapse at β = 0; 2 fast-stress cases | Development (Notebook 08), final test |
| Seconds-onset Lapse (`lapse_seconds_cases`, 5) | Lapse `forced_seconds` at β 0.5–3, and at β = 0 | Final test (33 cases in total) |
| Full grid (`full_cases`, 7,812) | 7,776 fatigue conditions: {Markov, FALCON, Lapse} × β {0.5, 1, 2, 3} × 4 distributions × recovery {0.2, 0.35, 0.5} × signal {0.5, 1, 2} × verify cost {0.25, 0.35, 0.5} × two-choice cost {0.5, 0.6, 0.75} × {150, 300} steps. Plus 36 zero-fatigue conditions: None and Lapse β = 0 over the cost and length axes | Controller selection on validation people; descriptive rates on test people |

### 9.5 Robustness (RQ4)

Plan with Markov β = 2, then test on β = 1 and 3, on 2× faster and 2× slower fatigue, on the FALCON world and on the Lapse world (QMDP, greedy, blind, verify). The deployable QMDP is the second robustness test, run in every case.

### 9.6 Statistics

- **Development:** paired bootstrap 95% intervals (2,000 resamples, seed 42) on matched sessions (Stage 1) or matched seeds (Stage 2). They are pointwise and unadjusted, and they are used for diagnosis and selection.
- **Final test** (fixed in `outputs/final_test/settings.json` before any run):
  - paired bootstrap with 4,000 resamples
  - two-sided sign-flip permutation p-values (exact for ≤ 20 pairs)
  - Holm correction per rival across the cases of each family (Stage 1), and per rival across the 12 Stage-2 cells, with the 12 primary comparisons as one family
- **Stage-2 primary rivals** (named in advance from development): no fatigue → fatigue-blind; Markov β = 1, 2 → greedy plus rest; FALCON → greedy; Lapse (answers) → fatigue-blind; Lapse (seconds) → two-choice.

### 9.7 Success criteria (fixed in advance; original §10.6)

- **H1:** at β̄ = 0 the main controller is within 1 SE of the best required baseline (no harm).
- **H2:** at β̄ ≥ 1 (Markov and FALCON) it beats every required baseline B1–B8 on net reward. Development test: the paired 95% CI is above zero. Final test: each Holm-adjusted p < 0.05 with a positive difference.
- **H3:** the planner beats the greedy controller B9, with the gap **growing with β̄**.
- **H4:** in Stage 2 it needs fewer human seconds to reach the target accuracy.

All results are reported, including nulls; the no-fatigue case is always shown.

### 9.8 Additional development checks (scripts, development data only)

| Check | Question | Result in one line |
|---|---|---|
| λ sweep (`run_lambda_sweep.py`) | Which price of time? | λ = 1.0 by the pre-set rule; QMDP flat for λ = 0.625–1.25 |
| E3 (`run_e3_ablation.py`) | Pausing or format choice? | Pausing gives half to all of QMDP's lead over verify; format choice is small and never clear |
| Pause break-even (`pause_break_even.py`) | Why does QMDP pause rarely? | At λ = 1 it needs ≥ 95% certainty that F ≥ 1; 10 s is near break-even (10.4 s) |
| E4 (`run_e4_headroom.py`) | Why does H3 fail? | The oracle's headroom over greedy is flat in β at λ = 1; FALCON is a ceiling |
| E5 (`run_e5_pause_length.py`) | Does the pause length matter? | 5 s: QMDP beats greedy plus rest; 20 s: QMDP stops pausing in Stage 2 and falls behind |
| Lapse seconds (`run_lapse_seconds_check.py`) | Is the Lapse loss an artefact of counting the onset in answers? | Mostly; counting in seconds shrinks the gap but does not remove it |
| E1b (`run_lapse_lambda_check.py`) | Is the Lapse loss caused by λ? | Answers world: a throughput effect of λ; seconds world: present at every λ, smallest near λ = 1 |

### 9.9 Figures

The planned figures and where they are:
- Cost–information frontier: Notebook 08, §9.
- Accuracy vs human seconds: Notebooks 10 and 11.
- Gain over the best fixed policy across conditions: Notebook 08 sensitivity plots; full-grid rates, figure 7 of the final test.
- Action mix vs fatigue: Notebooks 05–07.
- Robustness bars: Notebook 08.
- Simulator validation panel: Notebook 03.

The final-test figures 1–8 are produced by `scripts/make_final_test_figures.py`, together with `tables.md`.

---

## 10. Results summary

All numbers are at λ = 1.0 with the frozen code. "Clearly" means the interval excludes 0 (development) or the Holm-adjusted p < 0.05 (final test).

### 10.1 Hypotheses

| | Development (validation people) | Final test (test people, Holm) |
|---|---|---|
| **H1** | Holds in both stages | Holds (difference 0.0: QMDP makes the same choices as fatigue-blind) |
| **H2** | Markov 16/17, FALCON 1/3; full grid: Markov 93%, FALCON 55% of conditions | Markov 16/17 (miss: signal 0.5), FALCON 1/3, Lapse 0/8 |
| **H3** | Fails (QMDP − greedy +50.2, +46.8, +42.9, +42.2 at β 0.5–3) | Fails (slope −2.70 per β, 95% CI [−4.50, −1.03]) |
| **H4** | Not supported against the strongest baseline | No clear time-to-target difference (few complete pairs at 80% and in Lapse) |
| **Stage 2 vs named rival** | 0/10 groups clearly better, 2/10 clearly worse (Lapse) | 0/12 clearly better, 4/12 clearly worse |

### 10.2 Main findings

- **Stage 1 (information per second).** Where the fatigue model is right (Markov), tracking fatigue and pausing beats every required baseline and greedy by a wide margin (final test: QMDP − greedy +39.8 to +50.0). The lead does not grow with β, because the oracle headroom itself is flat at λ = 1 (E4). Robustness: QMDP stays clearly ahead of greedy in 5 of 6 misspecified worlds and is clearly behind in the FALCON world (−10.6).
- **FALCON** failures are a ceiling effect. In short sessions the decline barely starts, so even the oracle cannot beat always-verify (full grid: 16% of conditions pass at 150 steps, 93% at 300).
- **Lapse.** QMDP has no lapse state: it reads the lapse as mild fatigue, keeps asking, and weights the random answers as reliable. It is clearly behind always-verify in all 8 final-test Lapse cases. Counting the onset in seconds removes most of the Stage-2 gap, but not all of it.
- **Stage 2 (classifier accuracy).** Tracking fatigue clearly beats ignoring it in Markov (vs fatigue-blind +1.23 to +5.69). But a tuned **greedy-plus-rest rule matches QMDP** (Markov β = 2, 200 labels: −0.46 [−0.71, −0.20], Holm p = 0.030). The mechanism (200 labels): QMDP and greedy plus rest each collect about 2,100–2,300 answers at 2–4% error, against about 1,200 at 7–12% for fatigue-blind.
- **What QMDP does.** At λ = 1 it asks verify 98–99% of the time in Stage 2 and pauses on about 0.5% of turns. Pausing is its main lever (E3); its more accurate verify answers come from lower fatigue.
- **Human burden** (exploratory, defined after the test). QMDP interrupts tired people (F ≥ 3) 30–65% less than the never-pausing policies. It keeps them more fatigued than greedy plus rest, though (Markov β = 2: mean F 1.42 vs 0.53), because at λ = 1 it pauses only when very sure. It gets the most bits per human second, but fewer bits per question than the slower formats.
- **Development vs final test:** QMDP's differences against every rival agree closely (r = 0.999 in Stage 1, 0.968 in Stage 2).

**Supported conclusion.** Tracking hidden fatigue and pausing pays off in information per human second when fatigue behaves as the planner assumes. For classifier learning, resting is what matters, and a tuned rest rule gets the same benefit without belief tracking or planning. The approach is fragile under disengagement, and its human cost has to be discussed together with the price of a human second.

---

## 11. Limitations

1. **Simulated humans.** All results use a CIFAR-10H-calibrated simulator. The real data support baseline skill, speed and confusions, but not the assumed format-dependent fatigue, recovery or time inflation, and CIFAR-10H shows no average fatigue.
2. **Strong response-time signal.** Answers are about 6× slower at F = 4. Results shrink sharply at signal 0.5, which is not in Stage 2.
3. **Pause length** is a key assumption near break-even (E5). The Stage-2 look-ahead (remaining time / pause seconds) shortens with long pauses.
4. **Lapse modelling:** one never-ending lapse per session. The results depend on whether the onset is counted in answers or in seconds. No controller models disengagement.
5. **Planner knowledge:** the main QMDP and greedy use the true world's population model. The deployable QMDP shows the cost of not having it, which is large in FALCON (median −116 on the grid).
6. **Difficulty predictor** barely beats a constant, which weakens the instance-dependent part of the fatigue model.
7. **Gold checks:** the grid planner almost never chooses them, so its value-of-information role is untested. The DQN uses them heavily without benefit. The DQN had a modest training budget.
8. **Pool exhaustion:** always-verify exhausts the 4,000-image pool in 10 development runs; QMDP does in 4 (no fatigue).
9. **Statistics:** development intervals are pointwise and unadjusted. Time-to-target comparisons have few complete pairs. The exploratory burden metrics were defined after the final test. Final-test scoring uses 2,000 images (development: 1,000).

---

## 12. Implementation and repository

### 12.1 Layout (project folder `advanced_ml/`)

```
configs/     project_settings.json, classifier_settings.json, stage2_settings.json, frozen_src.json
data/        cifar10h/ (raw zip), cifar10/, pretrained/, splits/ (annotators, images, validation_roles),
             processed/ (development answers, simulator_calibration.npz, predicted_difficulty.csv), features/
src/         common.py (paths, run cache), review_checks.py (12 checks of the 19 Nov fixes)
  sim/       calib.py, validation.py, channel.py, fatigue.py, world.py, interactive.py
  control/   belief.py, qmdp.py, pomdp_grid.py, dqn.py, baselines.py
  learn/     classifier.py, losses.py, difficulty.py, al_loop.py
  eval/      stage1.py, stage2.py, study.py, stats.py, train_dqn.py
tests/       34 unit tests (stage-1 boundary, stage-2 learning, review fixes, seconds-onset lapse)
notebooks/   00–11
scripts/     λ sweep, E3, break-even, E4, E5, lapse checks, full grid, final test, figures, extra metrics
outputs/     every run in outputs/<stage>/<config-hash>/seed-<k>/ with its config.json
```

### 12.2 Provenance and freeze

- Every run folder stores its full configuration and a code fingerprint (sha256 of all `src/**/*.py`). Reruns reuse a cached run only if the configuration and the code match.
- `src/` was frozen on 21 November (22 files, `configs/frozen_src.json`). The full-grid, final-test and E1b scripts refuse to run on changed code. Notebooks 04–10 were rerun with the frozen code and reproduced the earlier λ = 1.0 results exactly.
- **Current result folders:**
  - `study/cbf5a08c06` (Stage-1 initial study)
  - `stage2_study/1e2d6d55c0` (Stage-2 development)
  - `dqn_training/5453e3e198`
  - `full_grid/2954f9bca0` (selection)
  - `final_test/` (with `figures/` and `extra_metrics/`)
- **Known leftovers kept to preserve the fingerprint:**
  - The Stage-2 options label "QMDP provisional from Notebook 08; full Stage-1 grid pending".
  - Unused `stored_action` and `falcon_quality`, and unused imports.
  - A comment in `full_cases()` about "rare disengagement" at β = 0. Lapse β = 0 now uses the forced lapse.

### 12.3 Rules followed (from the original brief)

1. No controller reads hidden simulator state (enforced by tests).
2. No tuning on test people or final-test images; the final test was run once.
3. Fixed seeds; every result saved with its configuration and fingerprint.
4. β = 0 / None always reported alongside the fatigue settings.
5. Paired differences with 95% intervals, not single runs.
6. Vectorised NumPy for Stage 1; PyTorch only for the classifier, the difficulty predictor and the DQN.

---

## 13. Literature map

Status: **Read** = text read directly (13 Nov); **Survey** = via our survey's summary; **Web** = abstract only.

| Paper | Use in this project | Status |
|---|---|---|
| Zhang et al., *Fatigue-Aware Learning to Defer via Constrained Optimisation (FALCON)*, arXiv 2604.00904 | FALCON fatigue curve and parameter ranges; Lagrangian budget view of λ; domain randomisation; stated future work (instance-dependent fatigue) = our extension; mammography validation curve (recall 78% → 66% over 100 readings) | Read |
| Banerjee et al., *To Ask or Not To Ask* (LinUCB-QG), ICRA 2025, arXiv 2405.06908 | Workload from recent query history; baselines B6, B7; bandit (myopic) contrast | Read |
| Singh, Roy, Chanel, *POMDP-Based Adaptive Interaction*, HHAI 2022 | POMDP over workload, belief update, point-based solvers → C2 | Read |
| Fatima, Alraeesi, Alzaabi, *Human-State-Aware Active Learning* (our survey) | Gap statement (no method jointly adapts format, timing and state; Sec. XI-C); simulator properties; CIFAR-10H / ImageNet-16H analysis | Read |
| Beck et al., CLARIFIER | Verify vs correct cost ratio 3–4× | Survey |
| Martín-Urcelay et al., *Beyond Labels*, arXiv 2602.15738 | Information-per-time objective; time models; baseline B8 | Survey |
| Liu et al., ALCL | Complementary labels for "no"; annotators agree with the model (yes-bias) | Survey |
| Hu et al., ALPF (ICLR 2019) | Partial-feedback questions (two-choice, "neither") | Survey |
| Simchi-Levi et al. (recovering bandits); Leqi et al. (rebounding bandits) | Recovery during inactivity → pause as rest | Survey |
| Chen et al. (trust-aware POMDP); Akash et al. | Learn the human model, then plan | Survey |
| Xue et al., arXiv 2604.12667 | Fatigue prediction and action masking | Survey |
| Peterson et al., CIFAR-10H (ICCV 2019) | Calibration data | Read (data analysed) |
| arXiv 2501.00277, *Efficient HIL Active Learning* | Joint question and data-point choice by information gain, no fatigue (related work) | Web |
| Mortágua, fatigue-aware annotator selection, arXiv 2507.23756 | +1 fatigue level per 20 annotations; selects annotators, not formats | Web / Survey |

**Novelty claim (re-check before submission):** to our knowledge, no published method jointly decides *whether* to query and *which format* to ask while planning over a *hidden, format-dependent fatigue state* in active learning.

---

## 14. How the project developed (short timeline)

| Date | Event |
|---|---|
| 13 Nov | Restart from scratch after the dead ends in §3.1; brief written; notebooks 00–10 built and executed at λ = 0.25 |
| 19 Nov | Code review (16 findings) |
| 20 Nov | Fixes rerun at λ = 0.25 (QMDP kept its Stage-1 lead but lost its Stage-2 advantage); teammate's λ = 0.75 check |
| 21 Nov | λ sweep → λ = 1.0; Stage-2 rest thresholds tuned per setting; seconds-onset Lapse; E3; code frozen |
| 21–22 Nov | Full grid (QMDP selected); E4, E5, break-even test; notebooks 04–10 rerun on the frozen code |
| 22 Nov | Final test run once; notebook 11; exploratory metrics; E1b |

The superseded 13 November numbers (for example the Stage-2 "47% less time to 75%" against greedy, or Markov β = 2 QMDP 145.69 bits in 174.39 s) came from the pre-review simulator at λ = 0.25 and should not be cited as results.

---

## 15. Open items

### 15.1 Assumptions to confirm with the professor (no confirmation recorded)

- "Binomial" fatigue = Bernoulli per-question increments (Binomial counts), plus the Bernoulli-mixture person variant. Both are included.
- Simulation-only evaluation is acceptable, given that no real multi-format fatigue dataset exists.
- The success criteria in §9.7, and the 1-pp practical threshold from the old study (Stage 2 reports gains in accuracy points at fixed budgets).

### 15.2 Optional extras not done

ImageNet-16H (validate the simulator on a second dataset), dopanim, a top-3 question as a fifth format, the Bayesian soft-label ablation, B12, and SARSOP / MOMDP solvers for C2. These belong under future work, together with real-human evaluation, a lapse state in the controller, and better fatigue tracking (for example with gold checks).

---

## Appendix A. Original brief vs implementation

| Original brief | Implementation (and why) |
|---|---|
| Title "ProjectBrief.md", status "restarting 13 Nov", deadline plan | Project complete; this brief |
| §5.2 "don't ask" costs τ_skip = 0.1 relative time (≈ 0.16 s) | Pause = 10 s (D11) |
| §5.5 λ initial 0.25, sweep or fit by bisection | λ = 1.0 by a pre-set sweep rule (D10) |
| §6.4 P(up) = u_a(1 + η(1 − p̂_x)) | Same form × fatigue_scale 0.1 × per-person rate U(0.5, 1.5) (D12) |
| §6.5 FALCON: warm-up + decline, logit shift from the peak, optional recovery | Fatigue from the start of the decline, clipped to [0, 4], warm-up treated as practice, peak ≤ 0.99, no recovery by default; planner chain from 20,000 curves (D13) |
| §6.5 M-Lapse: engaged/disengaged chain σ(α₀ + α₁F) with recovery r_L, calibrated to 0.6% | Forced lapse once per session, in answers or seconds, no recovery, independent of F; the 0.6% hazard is the legacy `rare` mode only (D14) |
| §6.5 lapsed verify: yes with 0.5 + κ' | All formats uniformly random (verify 50/50) |
| §6.6 controller uses μ̂_j, σ̂_j | Population timing plus the SpeedTracker personal offset; σ_pop for scaling (D16) |
| §6.8 recovery r ~ U(0.2, 0.5) per person | r ± U(−0.15, 0.15), i.e. U(0.2, 0.5) at r = 0.35 (consistent) |
| §7 Bayesian soft-label option | Not implemented |
| §8.1 QMDP re-solved every R steps | Re-solved after every retrain in Stage 2 (every 10 answers), with a receding horizon |
| §8.2 grid planner with interpolation, SARSOP as alternative | Nearest-grid projection, 16 contexts, one-step look-ahead with the real image; no SARSOP |
| §8.3 DQN trained on Stage 1, then fine-tuned in Stage 2 | Stage 1 only; training strengthened (48k updates, γ = 0.99) |
| §9.2 B9 myopic, B10 oracle, B11 random image, B12 old schedule | All but B12 implemented; greedy plus rest and deployable QMDP added (D15) |
| §10.2 paired differences against B3 and B8 and the best fixed policy | Against all nine rivals; the primary rival per setting named in advance; Holm on the final test |
| §10.3 Stage 2 on a reduced grid | As planned, plus Lapse with the onset in seconds on the final test |
| §11 prototype tables and 13 Nov checkpoints (26 conditions, 13 policies, 800 runs, λ = 0.25, "zero answers after a rare lapse", outputs/study/06a07e43ac, stage2_study/ea58267cbc) | Superseded: 28 (+5) cases, 15 Stage-1 and 10 Stage-2 policies, 1,000 development runs, forced lapses (130,635 answers after a lapse in the development study), results in §10 |
| §12 repository `aml-hsal/` with annotator.py, plots.py, yaml configs, prototypes/ | Actual layout in §12.1 (answer model in channel.py and world.py; JSON configs; no prototypes folder) |
