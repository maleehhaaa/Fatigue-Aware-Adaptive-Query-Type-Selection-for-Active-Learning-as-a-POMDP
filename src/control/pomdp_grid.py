"""Finite-horizon planning on a 0.1-spaced grid of fatigue beliefs."""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree
from src.control.belief import binned_observations, binned_observations_batch


def belief_grid(resolution=10):
    return np.array([row for row in product(range(resolution + 1), repeat=5)
                     if sum(row) == resolution], dtype=float) / resolution


def context_features(contexts):
    top = np.sort(contexts['probabilities'], axis=1)[:, -2:]
    return np.column_stack([top, contexts['difficulty'] / 4])


class GridPlanner:
    def __init__(self, model, context_count=16, resolution=10):
        self.model = model
        self.beliefs = belief_grid(resolution)
        self.belief_tree = cKDTree(self.beliefs)
        all_features = context_features(model.contexts)
        # Spread representatives along confidence; all four classifier checkpoints contribute.
        order = np.argsort(all_features[:, 1])
        self.representatives = order[np.linspace(0, len(order) - 1, context_count).astype(int)]
        self.context_tree = cKDTree(all_features[self.representatives])
        self.context_lookup = self.context_tree.query(all_features)[1]
        count, beliefs = context_count, len(self.beliefs)
        self.actions = np.zeros((model.scenario.steps, count, beliefs), dtype=np.uint8)
        self.values = np.zeros((model.scenario.steps + 1, beliefs))
        destinations = np.zeros((count, beliefs, 5, 10), dtype=np.int32)
        observation_chance = np.zeros((count, beliefs, 5, 10))
        expected_reward = np.einsum('bf,cfa->cba', self.beliefs, model.reward[self.representatives])
        for context, index in enumerate(self.representatives):
            for action in range(5):
                if action == 0:
                    likelihood = np.ones((5, 1))
                else:
                    likelihood = binned_observations(model.correct[index, :, action],
                        model.scenario.signal, .4 / model.population['sigma'], gold=action == 4)
                joint = self.beliefs[:, :, None] * likelihood[None, :, :]
                chances = joint.sum(1)
                posterior = joint / np.maximum(chances[:, None, :], 1e-30)
                next_belief = np.einsum('bfo,fg->bog', posterior, model.transition[index, :, action, :])
                nearest = self.belief_tree.query(next_belief.reshape(-1, 5))[1].reshape(beliefs, -1)
                destinations[context, :, action, :likelihood.shape[1]] = nearest
                observation_chance[context, :, action, :likelihood.shape[1]] = chances
        for step in range(model.scenario.steps - 1, -1, -1):
            future = (observation_chance * self.values[step + 1][destinations]).sum(3)
            values = expected_reward + future
            self.actions[step] = values.argmax(2)
            self.values[step] = values.max(2).mean(0)

    def action(self, contexts, indices, belief, step):
        """One-step look-ahead on the grid's value function, using the actual picture.

        Review fix: the old rule returned the action stored for the nearest of 16 representative
        pictures, so it never used the real picture's information and time.
        """
        model, count = self.model, len(indices)
        # Start from the nearest grid belief, exactly as in the backward pass; otherwise a pause,
        # which barely moves the belief, rounds back to "fresh" and looks like a free reset.
        belief = self.beliefs[self.belief_tree.query(belief)[1]]
        scores = np.zeros((count, 5))
        wrong_shift = .4 / model.population['sigma']
        for action in range(5):
            reward = np.einsum('nf,nf->n', belief, model.reward[indices, :, action])
            if action == 0:
                likelihood = np.ones((count, 5, 1))
            else:
                likelihood = binned_observations_batch(model.correct[indices, :, action],
                    model.scenario.signal, wrong_shift, gold=action == 4)
            joint = belief[:, :, None] * likelihood
            chances = joint.sum(1)
            posterior = joint / np.maximum(chances[:, None, :], 1e-30)
            following = np.einsum('nfo,nfg->nog', posterior, model.transition[indices, :, action, :])
            nearest = self.belief_tree.query(following.reshape(-1, 5))[1].reshape(count, -1)
            scores[:, action] = reward + (chances * self.values[step + 1][nearest]).sum(1)
        return scores.argmax(1)

    def stored_action(self, contexts, indices, belief, step):
        """The original lookup rule, kept for comparison."""
        context = self.context_lookup[indices]
        state = self.belief_tree.query(belief)[1]
        return self.actions[step, context, state].astype(int)
