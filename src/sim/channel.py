"""Answer probabilities and information scores. Supports batches of pictures."""
import numpy as np
from src.common import sigmoid

EASE = np.array([0, 1.5, 1, 0, 1.5])
WEIGHT = np.array([0, .25, .5, 1, .25])
ACTION_NAMES = ['skip', 'verify', 'two_choice', 'full', 'gold']


def answer_channel(probabilities, difficulty, fatigue, action, beta, skill, confusion, yes_bias=.04, lapse=None):
    """Rows are possible true classes; columns are possible answers (padded to 10)."""
    probabilities = np.atleast_2d(probabilities)
    count = len(probabilities)
    action = np.broadcast_to(action, (count,))
    fatigue = np.broadcast_to(fatigue, (count,))
    accuracy = sigmoid(np.asarray(difficulty) + skill + EASE[action] - np.asarray(beta) * WEIGHT[action] * fatigue)
    top = np.argsort(-probabilities, axis=1)[:, :2]
    confusion = np.broadcast_to(confusion, (count, 10, 10))
    channel = np.zeros((count, 10, 10))
    for kind in [1, 2, 3, 4]:
        rows = np.flatnonzero(action == kind)
        if not len(rows):
            continue
        acc = accuracy[rows, None]
        if kind == 3:
            channel[rows] = (1 - acc[:, :, None]) * confusion[rows]
            channel[rows[:, None], np.arange(10), np.arange(10)] = acc
        elif kind in [1, 4]:
            yes = np.broadcast_to(np.clip(1 - acc + yes_bias * fatigue[rows, None], 0, 1), (len(rows), 10)).copy()
            yes[np.arange(len(rows)), top[rows, 0]] = acc[:, 0]
            channel[rows, :, 0] = yes
            channel[rows, :, 1] = 1 - yes
        else:
            channel[rows, :, 2] = acc
            first_weight = confusion[rows[:, None], np.arange(10), top[rows, 0, None]]
            second_weight = confusion[rows[:, None], np.arange(10), top[rows, 1, None]]
            weight_sum = first_weight + second_weight
            first_share = np.divide(first_weight, weight_sum, out=np.full_like(first_weight, .5), where=weight_sum > 0)
            channel[rows, :, 0] = (1 - acc) * first_share
            channel[rows, :, 1] = (1 - acc) * (1 - first_share)
            for position in [0, 1]:
                channel[rows, top[rows, position], :] = 0
                channel[rows, top[rows, position], position] = acc[:, 0]
                channel[rows, top[rows, position], 1 - position] = (1 - acc[:, 0]) / 2
                channel[rows, top[rows, position], 2] = (1 - acc[:, 0]) / 2
    # A skip has one uninformative outcome, ensuring every row sums to one.
    channel[action == 0, :, 0] = 1
    if lapse is not None:
        random_channel = np.zeros_like(channel)
        for kind, answers in [(0, 1), (1, 2), (2, 3), (3, 10), (4, 2)]:
            random_channel[action == kind, :, :answers] = 1 / answers
        channel = np.where(np.asarray(lapse)[:, None, None], random_channel, channel)
    return channel


def mutual_information(probabilities, channel):
    answer_probability = np.einsum('ny,nyz->nz', probabilities, channel)
    ratio = np.log2(channel.clip(1e-12) / answer_probability[:, None, :].clip(1e-12))
    return np.maximum(0, (probabilities[:, :, None] * channel * ratio).sum((1, 2)))


def correct_probability(probabilities, channel, action):
    """Expected semantic correctness; ordinary correctness is never an observation."""
    action = np.broadcast_to(action, (len(probabilities),))
    top = np.argsort(-probabilities, axis=1)[:, :2]
    target = np.broadcast_to(np.arange(10), (len(probabilities), 10)).copy()
    target[action == 1] = (np.arange(10)[None, :] != top[action == 1, :1]).astype(int)
    target[action == 2] = np.where(np.arange(10)[None, :] == top[action == 2, :1], 0,
        np.where(np.arange(10)[None, :] == top[action == 2, 1:2], 1, 2))
    target[action == 4] = (np.arange(10)[None, :] != top[action == 4, :1]).astype(int)
    chosen = np.take_along_axis(channel, target[:, :, None], axis=2)[:, :, 0]
    return (probabilities * chosen).sum(1)
