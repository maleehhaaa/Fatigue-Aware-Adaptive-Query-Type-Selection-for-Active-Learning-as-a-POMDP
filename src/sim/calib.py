"""Fit the simulated people using the saved participant and image splits."""
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from src.common import ROOT, sigmoid


def fit_calibration():
    with ZipFile(ROOT / 'data/cifar10h/cifar10h-raw.zip') as archive:
        raw = pd.read_csv(archive.open('cifar10h-raw.csv'))
    raw = raw.rename(columns={'cifar10_test_test_idx': 'image_id'})
    people = pd.read_csv(ROOT / 'data/splits/annotators.csv').sort_values('annotator_id')
    images = pd.read_csv(ROOT / 'data/splits/images.csv')
    development_ids = images.loc[images.image_split != 'final_test', 'image_id']
    records = raw[(raw.is_attn_check == 0) & raw.image_id.isin(development_ids)].copy()
    train_ids = people.loc[people.annotator_split == 'train', 'annotator_id']
    train = records[records.annotator_id.isin(train_ids)].copy()
    counts = train.groupby('image_id').correct_guess.agg(['sum', 'size'])
    success = (counts['sum'] + 0.5) / (counts['size'] + 1)
    difficulty = np.zeros(10000)
    difficulty[counts.index] = np.log(success / (1 - success))
    true_labels = images.sort_values('image_id').true_label.to_numpy()

    # Person skill is an intercept: compare their answers with image-based expectations.
    person_ids = people.annotator_id.to_numpy()
    theta, mu, sigma = [], [], []
    for person_id in person_ids:
        answers = records[records.annotator_id == person_id]
        d = difficulty[answers.image_id]
        target = (answers.correct_guess.sum() + 0.5) / (len(answers) + 1)
        theta.append(brentq(lambda skill: sigmoid(d + skill).mean() - target, -15, 15))
        log_time = np.log(answers.reaction_time.clip(200, 30000) / 1000)
        residual = log_time - 0.8 * (1 - sigmoid(d)) - 0.4 * (1 - answers.correct_guess)
        residual = residual + 0.06 * answers.trial_index / 100
        mu.append(residual.mean())
        sigma.append(max(residual.std(), 0.08))
    theta, mu, sigma = map(np.array, [theta, mu, sigma])
    train_mask = people.annotator_split.eq('train').to_numpy()
    theta_for_record = pd.Series(theta, index=person_ids).loc[train.annotator_id].to_numpy()
    offset = brentq(lambda shift: sigmoid(difficulty[train.image_id] + theta_for_record + shift).mean() - train.correct_guess.mean(), -3, 3)
    theta += offset

    # Wrong full labels follow observed image confusions; class confusions are the fallback.
    confusion = np.ones((10, 10)) * 0.1
    np.fill_diagonal(confusion, 0)
    wrong = train[train.correct_guess == 0]
    np.add.at(confusion, (wrong.true_label.to_numpy(), wrong.chosen_label.to_numpy()), 1)
    confusion /= confusion.sum(1, keepdims=True)
    image_errors = np.zeros((10000, 10))
    np.add.at(image_errors, (wrong.image_id.to_numpy(), wrong.chosen_label.to_numpy()), 1)
    empty = image_errors.sum(1) == 0
    image_errors[empty] = confusion[true_labels[empty]]
    image_errors /= image_errors.sum(1, keepdims=True)
    calibration = {'person_ids': person_ids, 'person_split': people.annotator_split.to_numpy(dtype='U12'),
        'theta': theta, 'mu': mu, 'sigma': sigma, 'difficulty': difficulty,
        'true_labels': true_labels, 'confusion': confusion, 'image_errors': image_errors,
        'public_skill': theta[train_mask].mean(), 'public_mu': mu[train_mask].mean(),
        'public_sigma': np.sqrt(np.mean(sigma[train_mask] ** 2) + np.var(mu[train_mask]))}
    np.savez_compressed(ROOT / 'data/processed/simulator_calibration.npz', **calibration)
    return calibration, train


def validate_simulator(calibration, train, seed=42):
    """Replay training-session image/order structure, with newly sampled answers."""
    random = np.random.default_rng(seed)
    ids = train.image_id.to_numpy()
    person = np.searchsorted(calibration['person_ids'], train.annotator_id)
    probability = sigmoid(calibration['difficulty'][ids] + calibration['theta'][person])
    correct = random.random(len(train)) < probability
    # Rare disengagement is calibrated to 0.6% within 200 questions.
    hazard = -np.log(1 - 0.006) / 200
    onset = random.exponential(1 / hazard, len(calibration['person_ids']))
    disengaged = train.trial_index.to_numpy() > onset[person]
    correct[disengaged] = random.random(disengaged.sum()) < 0.1
    log_time = (calibration['mu'][person] + 0.8 * (1 - sigmoid(calibration['difficulty'][ids]))
        + 0.4 * (~correct) - 0.06 * train.trial_index.to_numpy() / 100
        + random.normal(size=len(train)) * calibration['sigma'][person])
    simulated = train[['annotator_id', 'trial_index', 'image_id', 'true_label']].copy()
    simulated['correct_guess'] = correct.astype(int)
    simulated['response_seconds'] = np.exp(log_time)
    simulated['chosen_label'] = calibration['true_labels'][ids]
    wrong_rows = np.flatnonzero(~correct)
    error_probabilities = calibration['image_errors'][ids[wrong_rows]]
    wrong_answers = (random.random(len(wrong_rows))[:, None] > error_probabilities.cumsum(1)).sum(1).clip(0, 9)
    simulated.loc[simulated.index[wrong_rows], 'chosen_label'] = wrong_answers
    return simulated, onset
