"""Batched private simulator. Only its observation dictionary goes to controllers."""
import numpy as np
from src.common import sigmoid
from src.sim.channel import answer_channel, mutual_information
from src.sim.fatigue import beta_samples, falcon_fatigue, BASE_UP, FALCON_LOAD


class AnnotationWorld:
    def __init__(self, calibration, contexts, scenario, split='train', episodes=500, seed=42,
                 max_steps=None, practice_cap=None):
        self._calibration, self.contexts, self.scenario = calibration, contexts, scenario
        random = np.random.default_rng(seed)
        self.count, self.step_number = episodes, 0
        tape_steps = scenario.steps if max_steps is None else max_steps
        self._practice_cap = practice_cap
        self._practice_trials = np.zeros(episodes)
        available = np.flatnonzero(calibration['person_split'] == split)
        self._person = random.choice(available, episodes, replace=episodes > len(available))
        self._beta = beta_samples(random, scenario.beta, scenario.distribution, episodes)
        self._up_scale = random.uniform(.5, 1.5, episodes) * scenario.speed
        self._recovery = (scenario.recovery + random.uniform(-.15, .15, episodes)).clip(0, 1)
        self._fatigue = np.zeros(episodes)
        self._lapse = np.zeros(episodes, dtype=bool)
        self._workload = np.zeros(episodes)
        # Peak capped at 0.99 (was 1.0): a peak near 1 has unbounded log-odds.
        self._falcon = np.column_stack([random.uniform(.7, .9, episodes), random.uniform(.4, .5, episodes),
            random.uniform(.8, .99, episodes), random.uniform(.025, .1, episodes),
            random.uniform(.25, .5, episodes), random.uniform(.05, .1, episodes)])
        # Forced lapse: each session disengages after a random number of answered questions.
        # Drawn here, before the tapes, would shift every other random number, so it uses its own stream.
        self._lapse_onset = np.random.default_rng(seed + 900001).uniform(scenario.lapse_onset_min,
            scenario.lapse_onset_max, episodes)
        self.indices = random.integers(len(contexts['image_id']), size=(tape_steps + 1, episodes))
        # Separate random tapes keep policy-dependent actions from shifting the random stream.
        self._random = random.random((tape_steps, episodes, 4))
        self._noise = random.normal(size=(tape_steps, episodes))
        self.seconds = np.zeros(episodes)
        self.information = np.zeros(episodes)
        self.errors = np.zeros(episodes)
        self.questions = np.zeros(episodes)
        self.action_counts = np.zeros((episodes, 5), dtype=int)
        self.last = {}

    def oracle_fatigue(self):
        if self.scenario.model == 'FALCON' and self.scenario.beta > 0:
            return falcon_fatigue(self._workload, self._falcon, self.scenario.steps)
        if self.scenario.model == 'None' or self.scenario.beta == 0:
            return np.zeros(self.count)
        return self._fatigue.copy()

    def step(self, actions):
        step, scenario = self.step_number, self.scenario
        actions = np.asarray(actions, dtype=int)
        indices = self.indices[step]
        probabilities = self.contexts['probabilities'][indices]
        image_ids = self.contexts['image_id'][indices]
        calibration, person = self._calibration, self._person
        true_class = calibration['true_labels'][image_ids]
        difficulty = calibration['difficulty'][image_ids]
        fatigue = self._fatigue.copy()
        if scenario.model == 'None' or scenario.beta == 0:
            fatigue[:] = 0
        elif scenario.model == 'FALCON':
            fatigue = falcon_fatigue(self._workload, self._falcon, scenario.steps)
        private_confusion = np.broadcast_to(calibration['confusion'], (self.count, 10, 10)).copy()
        private_confusion[np.arange(self.count), true_class] = calibration['image_errors'][image_ids]
        channel = answer_channel(probabilities, difficulty, fatigue, actions, self._beta,
            calibration['theta'][person], private_confusion,
            yes_bias=0 if scenario.beta == 0 else .04, lapse=self._lapse)
        actual_answers = channel[np.arange(self.count), true_class]
        answer = (self._random[step, :, 0, None] > actual_answers.cumsum(1)).sum(1).clip(0, 9)
        top = np.argsort(-probabilities, axis=1)[:, :2]
        target = true_class.copy()
        target[actions == 1] = (true_class[actions == 1] != top[actions == 1, 0]).astype(int)
        target[actions == 2] = np.where(true_class[actions == 2] == top[actions == 2, 0], 0,
            np.where(true_class[actions == 2] == top[actions == 2, 1], 1, 2))
        target[actions == 0] = 0
        correct = answer == target
        gold = actions == 4
        gold_accuracy = sigmoid(4 + calibration['theta'][person] + 1.5 - self._beta * .25 * fatigue)
        correct[gold] = self._random[step, gold, 0] < np.where(self._lapse[gold], .5, gold_accuracy[gold])
        answer[gold] = (~correct[gold]).astype(int)
        effective_difficulty = np.where(gold, 4, difficulty)
        practice = step if self._practice_cap is None else np.minimum(self._practice_trials, self._practice_cap)
        log_seconds = (calibration['mu'][person] + .8 * (1 - sigmoid(effective_difficulty))
            + np.log(scenario.times[actions]) + scenario.signal * float(calibration['public_sigma']) * fatigue
            + .4 * (~correct) - .06 * practice / 100 + self._noise[step] * calibration['sigma'][person])
        seconds = np.exp(log_seconds)
        seconds[actions == 0] = scenario.pause_duration(calibration['public_mu'])
        information = mutual_information(probabilities, channel)
        information[(actions == 0) | gold] = 0
        active = (actions > 0) & ~gold
        self.seconds += seconds
        self.information += information
        self.errors += active & ~correct
        self.questions += active
        self.action_counts[np.arange(self.count), actions] += 1
        self.last = {'fatigue': fatigue, 'correct': correct, 'information': information,
            'reward': information - scenario.lam * seconds, 'seconds': seconds}

        # Generate observations before changing fatigue: the answer came from the old state.
        observations = {'answer': answer, 'seconds': seconds, 'gold': gold,
            'gold_correct': np.where(gold, correct, False)}
        if scenario.model == 'None' or scenario.beta == 0:
            self._fatigue[:] = 0
        else:
            up = BASE_UP[actions] * scenario.fatigue_scale
            up = (up * self._up_scale * (2 - sigmoid(effective_difficulty))).clip(0, 1)
            increase = (actions > 0) & (self._random[step, :, 1] < up)
            decrease = (actions == 0) & (self._random[step, :, 1] < self._recovery)
            self._fatigue = (self._fatigue + increase - decrease).clip(0, 4)
            load = FALCON_LOAD[actions] * (2 - sigmoid(effective_difficulty))
            self._workload += load
            if scenario.falcon_recovery:
                self._workload = np.maximum(0, self._workload - (actions == 0) * scenario.recovery)
        if scenario.model == 'Lapse' and scenario.lapse_mode == 'forced':
            # Review fix: the rare hazard produced almost no lapses, so the setting duplicated
            # Markov. Now every session disengages once, after U(lapse_onset_min, lapse_onset_max)
            # answered questions.
            self._lapse |= (self.questions + (actions == 4)) >= self._lapse_onset
        elif scenario.model == 'Lapse' and scenario.lapse_mode == 'forced_seconds':
            # Added 21 Nov: the same forced lapse, but the onset is counted in human seconds since
            # the session began (pauses included), so it no longer depends on how fast a policy asks.
            # Same onset draw and same random stream as 'forced'.
            self._lapse |= self.seconds >= self._lapse_onset
        elif scenario.model == 'Lapse':
            # Legacy 'rare' mode: calibrated to ~0.6% of people within 200 questions.
            hazard = -np.log(1 - .006) / 200
            onset = (hazard * np.exp(scenario.lapse_dependence * fatigue)).clip(0, 1)
            self._lapse |= self._random[step, :, 2] < onset
        self.step_number += 1
        self._practice_trials += actions > 0
        return observations

    def lapsed(self):
        """Evaluation-only: which sessions are disengaged now."""
        return self._lapse.copy()

    def results(self):
        return {'information': self.information.copy(), 'seconds': self.seconds.copy(),
            'reward': self.information - self.scenario.lam * self.seconds,
            'error_rate': self.errors / np.maximum(self.questions, 1),
            'questions': self.questions.copy(), 'action_counts': self.action_counts.copy()}
