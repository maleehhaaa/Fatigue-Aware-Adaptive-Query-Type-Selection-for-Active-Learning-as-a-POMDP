"""Population-model planning tables. Inputs contain no simulator ground truth."""
import numpy as np
from src.common import sigmoid
from src.sim.channel import answer_channel, mutual_information, correct_probability
from src.sim.fatigue import transition_table


class PlanningModel:
    def __init__(self, contexts, scenario, population, plan=True, planning_indices=None):
        self.contexts, self.scenario, self.population = contexts, scenario, population
        probabilities, difficulty = contexts['probabilities'], contexts['difficulty']
        self.transition = transition_table(difficulty, scenario)
        count = len(probabilities)
        self.information = np.zeros((count, 5, 5))
        self.correct = np.ones((count, 5, 5))
        self.seconds = np.zeros((count, 5, 5))
        for fatigue in range(5):
            for action in range(1, 5):
                channel = answer_channel(probabilities, difficulty, fatigue, action,
                    scenario.beta, population['skill'], population['confusion'],
                    yes_bias=0 if scenario.beta == 0 else .04)
                self.correct[:, fatigue, action] = correct_probability(probabilities, channel, action)
                if action != 4:
                    self.information[:, fatigue, action] = mutual_information(probabilities, channel)
                else:
                    # Gold item is a separate easy, known-answer verification question.
                    self.correct[:, fatigue, action] = sigmoid(4 + population['skill'] + 1.5 - scenario.beta * .25 * fatigue)
            log_base = population['mu'] + .8 * (1 - sigmoid(difficulty))
            for action in range(5):
                if action == 0:
                    self.seconds[:, fatigue, action] = scenario.pause_duration(population['mu'])
                else:
                    base = np.full(count, population['mu'] + .8 * (1 - sigmoid(4))) if action == 4 else log_base
                    chance_wrong = 1 - self.correct[:, fatigue, action]
                    # Expected seconds for the correct/wrong lognormal mixture.
                    self.seconds[:, fatigue, action] = (np.exp(base + .5 * population['sigma'] ** 2
                        + scenario.signal * population['sigma'] * fatigue) * scenario.times[action]
                        * (1 + chance_wrong * (np.exp(.4) - 1)))
        self.reward = self.information - scenario.lam * self.seconds
        # Greedy-plus-rest baseline: pause when expected fatigue reaches this level (tuned on training people).
        self.rest_threshold = 2.0
        self.value = np.zeros((scenario.steps + 1, 5))
        planned_reward = self.reward if planning_indices is None else self.reward[planning_indices]
        planned_transition = self.transition if planning_indices is None else self.transition[planning_indices]
        for step in range(scenario.steps - 1, -1, -1) if plan else []:
            future = np.einsum('cfag,g->cfa', planned_transition, self.value[step + 1])
            # QMDP omits gold: under its shortcut it does not pay to learn the state.
            self.value[step] = (planned_reward[:, :, :4] + future[:, :, :4]).max(2).mean(0)

    def action(self, indices, belief, step, name='qmdp'):
        rewards = self.reward[indices]
        if name == 'blind':
            return rewards[:, 0, :4].argmax(1)
        if name == 'myopic':
            return np.einsum('nf,nfa->na', belief, rewards[:, :, :4]).argmax(1)
        if name == 'rest_threshold':
            # Review fix: greedy almost never pauses, so "QMDP vs greedy" also measured resting.
            # This baseline is greedy, but pauses whenever expected fatigue is high.
            greedy = np.einsum('nf,nfa->na', belief, rewards[:, :, :4]).argmax(1)
            return np.where(belief @ np.arange(5) >= self.rest_threshold, 0, greedy)
        future = np.einsum('nfag,g->nfa', self.transition[indices], self.value[step + 1])
        values = (rewards + future)[:, :, :4]
        return np.einsum('nf,nfa->na', belief, values).argmax(1)
