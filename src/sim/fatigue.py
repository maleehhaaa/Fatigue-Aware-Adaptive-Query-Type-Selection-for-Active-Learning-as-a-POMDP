"""Different assumed fatigue stories; these are hypotheses, not measured laws."""
from dataclasses import dataclass, asdict, replace
from typing import Optional
import numpy as np
from src.common import sigmoid

# Chance per question of moving up one fatigue level, before scaling (skip, verify, 2-choice, full, gold).
BASE_UP = np.array([0, .06, .12, .25, .06])
# FALCON workload added per question (full label = 1), before the image-difficulty factor.
FALCON_LOAD = np.array([0, .25, .5, 1, .25])
LEGACY_PAUSE_FACTOR = .1  # Old skip duration: 0.1 x exp(population log-time mean), about 0.16 s.


@dataclass(frozen=True)
class Scenario:
    model: str = 'Markov'
    beta: float = 1.0
    distribution: str = 'constant'
    recovery: float = .35            # chance that one pause lowers fatigue by one level
    signal: float = 1.0
    verify_time: float = .35
    choice_time: float = .6
    speed: float = 1.0
    steps: int = 150
    lam: float = 1.0                 # was .25; chosen by scripts/run_lambda_sweep.py (outputs/lambda_sweep/decision.json)
    lapse_dependence: float = 0.0    # only used by the legacy 'rare' lapse mode
    falcon_recovery: bool = False
    # Review fixes (19 Nov 2026). Defaults are the realistic setting; see fast_stress_test().
    fatigue_scale: float = .1        # 0.1: ~190 full labels from fresh to F=4; 1.0: old ~15
    pause_seconds: Optional[float] = 10.  # seconds a "don't ask now" pause takes; None = old 0.16 s
    lapse_mode: str = 'forced'       # 'forced': one lapse per session; 'rare': old 0.6%/200 hazard;
                                     # 'forced_seconds' (added 21 Nov): one lapse per session, timed in seconds
    # Forced mode: the lapse starts after U(lapse_onset_min, lapse_onset_max) answered questions.
    # 'forced_seconds' mode: the same two numbers are human seconds since the session began
    # (pauses count as time). The answer-count 'forced' mode is unchanged.
    # Two plain numbers (not a tuple) so every scenario setting fits in one table column.
    lapse_onset_min: float = 20.
    lapse_onset_max: float = 100.

    def dictionary(self):
        return asdict(self)

    @property
    def times(self):
        return np.array([.1, self.verify_time, self.choice_time, 1., self.verify_time])

    def pause_duration(self, public_mu):
        """Seconds for action 0. The old setting made a pause cost about 0.16 s."""
        if self.pause_seconds is None:
            return LEGACY_PAUSE_FACTOR * float(np.exp(public_mu))
        return float(self.pause_seconds)


def fast_stress_test(scenario):
    """The pre-review simulator: exhaustion in ~15 full labels and near-free rest."""
    return replace(scenario, fatigue_scale=1., pause_seconds=None, lapse_mode='rare')


def deployable_assumption(scenario):
    """What a deployed planner could assume without knowing the true world.

    Costs, horizon, lambda and pause length are design choices, so they are copied. The fatigue
    family, strength, recovery, rate and signal are unknown, so fixed defaults are used.
    """
    return Scenario(model='Markov', beta=1., distribution='constant', recovery=.35, signal=1.,
        verify_time=scenario.verify_time, choice_time=scenario.choice_time, steps=scenario.steps,
        lam=scenario.lam, pause_seconds=scenario.pause_seconds)


def beta_samples(random, mean, distribution, size):
    if distribution == 'constant':
        return np.full(size, mean)
    if distribution == 'uniform':
        return random.uniform(0, 2 * mean, size)
    if distribution == 'mixture':
        return (random.random(size) < .3) * mean / .3
    return random.gamma(2, mean / 2, size)


def transition_table(difficulty, scenario):
    """Population planning model. Never contains a person's hidden parameters."""
    if scenario.model == 'FALCON' and scenario.beta > 0:
        return falcon_transition_table(difficulty, scenario)
    count = len(difficulty)
    table = np.zeros((count, 5, 5, 5))  # picture, current fatigue, action, next fatigue
    for fatigue in range(5):
        for action in range(5):
            if scenario.model == 'None' or scenario.beta == 0:
                table[:, fatigue, action, fatigue] = 1
            elif action == 0:
                table[:, fatigue, action, max(fatigue - 1, 0)] += scenario.recovery
                table[:, fatigue, action, fatigue] += 1 - scenario.recovery
            else:
                up = BASE_UP[action] * scenario.fatigue_scale
                chance = np.clip(up * scenario.speed * (2 - sigmoid(4 if action == 4 else difficulty)), 0, 1)
                table[:, fatigue, action, min(fatigue + 1, 4)] += chance
                table[:, fatigue, action, fatigue] += 1 - chance
    return table


# Population means of the FALCON parameter ranges used by the world (CIFAR-100, Table 2;
# peak capped at 0.99 so its log-odds stay finite).
FALCON_RANGES = {'initial': (.7, .9), 'base': (.4, .5), 'peak': (.8, .99), 'warm': (.025, .1),
                 'middle': (.25, .5), 'slope': (.05, .1)}


def falcon_level_gaps(length, samples=20000, seed=0):
    """Average workload between fatigue levels, over a fixed sample of FALCON people.

    Uses the same parameter ranges as the world, so the planner matches the population on
    average without knowing any one person's curve. Returns, for each level k = 0..3, the chance
    that a person at level k ever reaches k + 1 and the mean workload it takes when they do.
    """
    random = np.random.default_rng(seed)
    draw = {key: random.uniform(*value, samples) for key, value in FALCON_RANGES.items()}
    base, peak, middle, slope = draw['base'], draw['peak'], draw['middle'], draw['slope']
    start = base + (peak - base) * sigmoid(slope * middle * length)
    start_logit = np.log(start / (1 - start))
    crossing = np.full((samples, 5), np.inf)
    for level in range(5):
        quality = sigmoid(start_logit - level)
        reachable = quality > base + 1e-9
        share = (quality[reachable] - base[reachable]) / (peak[reachable] - base[reachable])
        crossing[reachable, level] = np.maximum(middle[reachable] * length
            + np.log(1 / share - 1) / slope[reachable], 0)
    reach, gap = np.zeros(4), np.full(4, np.inf)
    for level in range(4):
        here = np.isfinite(crossing[:, level])
        onward = here & np.isfinite(crossing[:, level + 1])
        reach[level] = onward.sum() / max(here.sum(), 1)
        if onward.any():
            gap[level] = (crossing[onward, level + 1] - crossing[onward, level]).mean()
    return reach, gap


def falcon_transition_table(difficulty, scenario):
    """Five-state approximation to the FALCON decline, averaged over the world's people.

    Fatigue is measured from the start of the decline, exactly as in the world (falcon_fatigue),
    so F = 0 means workload 0. A question moves up one level with chance
    reach x load / gap, where gap is the mean workload between levels and reach the share of
    people who get there (falcon_level_gaps). Review fix: interpolating a single mean curve at
    whole levels made the planner expect fatigue far too slowly (E[F] 1.3 vs 2.2 after 150 full
    labels). Recovery only if falcon_recovery is set.
    """
    count = len(difficulty)
    table = np.zeros((count, 5, 5, 5))
    reach, gap = falcon_level_gaps(scenario.steps)
    for fatigue in range(5):
        for action in range(5):
            if action == 0:
                if scenario.falcon_recovery and fatigue > 0 and np.isfinite(gap[fatigue - 1]):
                    down = min(scenario.recovery / gap[fatigue - 1], 1)
                    table[:, fatigue, 0, fatigue - 1] = down
                    table[:, fatigue, 0, fatigue] = 1 - down
                else:
                    table[:, fatigue, 0, fatigue] = 1
                continue
            load = FALCON_LOAD[action] * (2 - sigmoid(4 if action == 4 else difficulty))
            if fatigue < 4 and np.isfinite(gap[fatigue]):
                up = np.clip(reach[fatigue] * load / gap[fatigue], 0, 1)
            else:
                up = np.zeros(count)
            table[:, fatigue, action, min(fatigue + 1, 4)] += up
            table[:, fatigue, action, fatigue] += 1 - up
    return table


def falcon_quality(workload, parameters, length):
    """FALCON Eq. 2 (warm-up + sigmoid decline). Kept for reference and plots."""
    initial, base, peak, warm, middle, slope = parameters.T
    before = initial + (peak - initial) * (workload / (warm * length)).clip(0, 1) ** 2
    after = base + (peak - base) * sigmoid(-slope * (workload - middle * length))
    return np.where(workload <= warm * length, before, after)


def falcon_fatigue(workload, parameters, length):
    """Fatigue in logit units, measured from the start of FALCON's decline phase.

    Review fix: the old version measured from the peak, so the warm-up made people start the
    session already "tired" (median F = 0.8) and a peak near 1 sent F to 12-14. Warm-up is now
    treated as practice (the response-time model already has a practice term), F starts at 0,
    never decreases without recovery, and is clipped to the 0-4 range the controllers use.
    """
    initial, base, peak, warm, middle, slope = parameters.T
    start = base + (peak - base) * sigmoid(slope * middle * length)
    now = base + (peak - base) * sigmoid(-slope * (workload - middle * length))
    fatigue = np.log(start / (1 - start)) - np.log(now / (1 - now))
    return fatigue.clip(0, 4)
