"""Infer fatigue from response time, without seeing normal-answer correctness."""
import numpy as np
from scipy.special import ndtr


def observation_likelihood(z, accuracy, signal, wrong_shift, gold_correct=None):
    centres = np.arange(5)[None, :] * signal
    fresh = np.exp(-.5 * (z[:, None] - centres) ** 2)
    wrong = np.exp(-.5 * (z[:, None] - centres - wrong_shift) ** 2)
    if gold_correct is None:
        return accuracy * fresh + (1 - accuracy) * wrong
    return np.where(gold_correct[:, None], accuracy * fresh, (1 - accuracy) * wrong)


def update_belief(belief, likelihood, transitions):
    posterior = belief * np.maximum(likelihood, 1e-30)
    posterior /= posterior.sum(1, keepdims=True)
    prediction = np.einsum('nf,nfg->ng', posterior, transitions)
    return prediction / prediction.sum(1, keepdims=True)


def binned_observations(accuracy, signal, wrong_shift, gold=False):
    # Five time bins cover both fresh and tired response times.
    edges = np.array([-np.inf, .5, 1.5, 2.5, 3.5, np.inf]) * max(signal, .5)
    centres = np.arange(5) * signal
    right = np.diff(ndtr(edges[None, :] - centres[:, None]), axis=1)
    wrong = np.diff(ndtr(edges[None, :] - centres[:, None] - wrong_shift), axis=1)
    if gold:
        return np.concatenate([accuracy[:, None] * right, (1 - accuracy[:, None]) * wrong], axis=1)
    return accuracy[:, None] * right + (1 - accuracy[:, None]) * wrong


def binned_observations_batch(accuracy, signal, wrong_shift, gold=False):
    """binned_observations for many pictures at once: accuracy is (n, 5) -> (n, 5, bins)."""
    edges = np.array([-np.inf, .5, 1.5, 2.5, 3.5, np.inf]) * max(signal, .5)
    centres = np.arange(5) * signal
    right = np.diff(ndtr(edges[None, :] - centres[:, None]), axis=1)
    wrong = np.diff(ndtr(edges[None, :] - centres[:, None] - wrong_shift), axis=1)
    accuracy = accuracy[:, :, None]
    if gold:
        return np.concatenate([accuracy * right[None], (1 - accuracy) * wrong[None]], axis=2)
    return accuracy * right[None] + (1 - accuracy) * wrong[None]


class SpeedTracker:
    """Learn each person's own speed from their first answers, shrunk toward the population.

    Review fix: the filter compared every response time with the population's average speed,
    so a naturally slow person looked permanently tired (one SD of personal speed is about half
    a fatigue level). The offset uses only response times, never hidden simulator state.
    """

    def __init__(self, count, prior_sd, noise_sd, answers=20):
        self.total = np.zeros(count)
        self.used = np.zeros(count)
        self.answers = answers
        self.shrink = (noise_sd / prior_sd) ** 2

    @property
    def offset(self):
        return self.total / (self.used + self.shrink)

    def update(self, residual, expected_shift, answered):
        use = answered & (self.used < self.answers)
        self.total[use] += residual[use] - expected_shift[use]
        self.used[use] += 1
