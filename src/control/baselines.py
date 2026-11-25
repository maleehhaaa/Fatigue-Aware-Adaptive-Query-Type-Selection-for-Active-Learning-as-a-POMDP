"""Small comparison rules. None receives a hidden human state."""
import numpy as np

POLICIES = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay',
            'gap_workload', 'blind', 'myopic', 'rest_threshold', 'qmdp', 'qmdp_fixed', 'grid', 'dqn', 'oracle']


def baseline_action(name, probabilities, belief, step, random):
    count = len(probabilities)
    if name in ['full', 'two_choice', 'verify']:
        return np.full(count, {'full': 3, 'two_choice': 2, 'verify': 1}[name])
    if name == 'random':
        return random.integers(1, 4, count)
    if name == 'alternating':
        return np.full(count, 1 + step % 3)
    if name == 'exp_decay':
        return np.where(random.random(count) < np.exp(-.01 * step), random.integers(1, 4, count), 0)
    top = np.sort(probabilities, axis=1)[:, -2:]
    uncertainty = 1 - (top[:, 1] - top[:, 0])
    # Review fix: an elementwise sum avoids spurious matmul warnings from the macOS math library.
    workload = (belief * np.arange(5)).sum(1)
    return np.where(uncertainty > .25 * workload, 3, np.where(uncertainty > .1, 1, 0))
