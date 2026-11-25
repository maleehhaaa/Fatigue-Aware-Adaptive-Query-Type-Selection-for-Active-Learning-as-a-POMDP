"""Uncertainty for paired comparisons; positive difference favours the first policy."""
import numpy as np


def paired_difference(table, first, second, seed=42, repeats=2000):
    wide = table.pivot(index='episode', columns='policy', values='reward')
    differences = (wide[first] - wide[second]).to_numpy()
    random = np.random.default_rng(seed)
    means = differences[random.integers(len(differences), size=(repeats, len(differences)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return {'first': first, 'second': second, 'difference': differences.mean(), 'ci_low': low, 'ci_high': high}
