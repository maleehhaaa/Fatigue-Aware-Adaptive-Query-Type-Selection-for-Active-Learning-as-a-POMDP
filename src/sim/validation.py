"""Descriptive simulator checks, with tolerances fixed before comparison.

Review note: these are internal consistency checks. The replay reuses the same people's fitted
skill and speed, and the lapse check samples from the configured hazard, so passing them does not
validate fatigue behaviour against independent data.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.common import sigmoid


def session_slopes(records, difficulty, shuffle=False, seed=42):
    random = np.random.default_rng(seed)
    slopes = []
    for _, person in records.groupby('annotator_id'):
        x = person.trial_index.to_numpy() / 100
        y = person.correct_guess.to_numpy() - sigmoid(difficulty[person.image_id])
        if shuffle:
            y = random.permutation(y)
        slopes.append(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2) * 100)
    return np.array(slopes)


def validation_checks(real, simulated, calibration, seed=42):
    real = real.copy()
    real['response_seconds'] = real.reaction_time.clip(200, 30000) / 1000
    def measures(records):
        time = records.response_seconds
        slopes = session_slopes(records, calibration['difficulty'])
        correct_time = time[records.correct_guess == 1].median()
        wrong_time = time[records.correct_guess == 0].median()
        return {'accuracy': records.correct_guess.mean(), 'median_seconds': time.median(),
            'log_time_sd': np.log(time).std(), 'wrong_correct_time_ratio': wrong_time / correct_time,
            'mean_slope_pp_per_100': slopes.mean(), 'slope_sd': slopes.std()}
    observed, predicted = measures(real), measures(simulated)
    checks = []
    tolerances = {'accuracy': .01, 'median_seconds': .15 * observed['median_seconds'],
        'log_time_sd': .15 * observed['log_time_sd'],
        'wrong_correct_time_ratio': .2 * observed['wrong_correct_time_ratio'],
        'mean_slope_pp_per_100': .5, 'slope_sd': .25 * observed['slope_sd']}
    for name in observed:
        checks.append({'check': name, 'real': observed[name], 'simulated': predicted[name],
            'allowed_absolute_difference': tolerances[name],
            'passed': abs(observed[name] - predicted[name]) <= tolerances[name]})
    def confusion(records):
        matrix = pd.crosstab(records.true_label, records.chosen_label).reindex(index=range(10), columns=range(10), fill_value=0).to_numpy(dtype=float, copy=True)
        np.fill_diagonal(matrix, 0)
        return matrix.ravel()
    correlation = spearmanr(confusion(real), confusion(simulated)).statistic
    checks.append({'check': 'confusion_rank_correlation', 'real': 1, 'simulated': correlation,
        'allowed_absolute_difference': .2, 'passed': correlation >= .8})
    random = np.random.default_rng(seed)
    hazard = -np.log(1 - .006) / 200
    rate = (random.exponential(1 / hazard, 100000) < 200).mean()
    checks.append({'check': 'lapse_onset_rate_200_trials', 'real': .006, 'simulated': rate,
        'allowed_absolute_difference': .0015, 'passed': abs(rate - .006) <= .0015})
    return pd.DataFrame(checks), real
