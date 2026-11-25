"""Run a documented initial study, or the full factorial grid when requested."""
from dataclasses import replace
from itertools import product
import json
import hashlib
import numpy as np
import pandas as pd
from src.common import ROOT, save_run
from src.sim.fatigue import Scenario, fast_stress_test
from src.eval.stage1 import benchmark
from src.eval.stats import paired_difference


def initial_cases():
    cases = [('no_fatigue', Scenario(model='None', beta=0), 'core')]
    for model, beta in product(['Markov', 'FALCON', 'Lapse'], [.5, 1., 2., 3.]):
        cases.append((f'{model}_beta_{beta:g}', Scenario(model=model, beta=beta), 'core'))
    base = Scenario(beta=2)
    for distribution in ['uniform', 'mixture', 'gamma']:
        cases.append((f'distribution_{distribution}', replace(base, distribution=distribution), 'sensitivity'))
    for recovery in [.2, .5]:
        cases.append((f'recovery_{recovery}', replace(base, recovery=recovery), 'sensitivity'))
    for signal in [.5, 2.]:
        cases.append((f'signal_{signal}', replace(base, signal=signal), 'sensitivity'))
    for verify, choice in [(.25, .6), (.5, .6), (.35, .5), (.35, .75)]:
        cases.append((f'cost_{verify}_{choice}', replace(base, verify_time=verify, choice_time=choice), 'sensitivity'))
    cases.append(('long_session_300', replace(base, steps=300), 'sensitivity'))
    cases.append(('lapse_without_fatigue', Scenario(model='Lapse', beta=0), 'control'))
    # Review fix: the pre-review simulator (exhaustion in ~15 full labels, ~0.16 s rest) is kept
    # as a labelled stress test, so the earlier results stay reproducible and comparable.
    for beta in [1., 2.]:
        cases.append((f'fast_stress_markov_beta_{beta:g}', fast_stress_test(Scenario(beta=beta)), 'stress'))
    return cases


# Added 21 Nov: Stage-1 Lapse cases with the onset counted in human seconds. Range: when always-verify
# reaches its 20th and 100th answer in the answer-count Lapse world (beta = 1, validation people):
# about 14 s and 88 s. initial_cases() and full_cases() are unchanged.
LAPSE_SECONDS_ONSET_STAGE1 = (14., 88.)


def lapse_seconds_cases():
    low, high = LAPSE_SECONDS_ONSET_STAGE1
    cases = [(f'Lapse_seconds_beta_{beta:g}', Scenario(model='Lapse', beta=beta, lapse_mode='forced_seconds',
              lapse_onset_min=low, lapse_onset_max=high), 'lapse_seconds') for beta in [.5, 1., 2., 3.]]
    cases.append(('lapse_seconds_without_fatigue', Scenario(model='Lapse', beta=0, lapse_mode='forced_seconds',
                  lapse_onset_min=low, lapse_onset_max=high), 'control'))
    return cases


def final_test_cases():
    """Stage-1 cases for the final test: the initial study plus the seconds-onset Lapse cases."""
    return initial_cases() + lapse_seconds_cases()


def full_cases():
    cases = []
    for steps, verify, choice in product([150, 300], [.25, .35, .5], [.5, .6, .75]):
        cases.append((f'full_{len(cases)}', Scenario(model='None', beta=0, steps=steps,
            verify_time=verify, choice_time=choice), 'full_grid'))
        # Rare disengagement remains possible at beta=0, so this is not a duplicate of None.
        cases.append((f'full_{len(cases)}', Scenario(model='Lapse', beta=0, steps=steps,
            verify_time=verify, choice_time=choice), 'full_grid'))
    axes = product(['Markov', 'FALCON', 'Lapse'], [.5, 1., 2., 3.],
        ['constant', 'uniform', 'mixture', 'gamma'], [.2, .35, .5], [.5, 1., 2.],
        [.25, .35, .5], [.5, .6, .75], [150, 300])
    for model, beta, distribution, recovery, signal, verify, choice, steps in axes:
        scenario = Scenario(model=model, beta=beta, distribution=distribution, recovery=recovery,
            signal=signal, verify_time=verify, choice_time=choice, steps=steps)
        cases.append((f'full_{len(cases)}', scenario, 'full_grid'))
    return cases


def run_study(contexts, dqn, full=False, episodes=500, seed=42):
    cases = full_cases() if full else initial_cases()
    policies = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay',
        'gap_workload', 'blind', 'myopic', 'rest_threshold', 'qmdp', 'qmdp_fixed', 'grid', 'dqn', 'oracle']
    tables = []
    for index, (name, scenario, purpose) in enumerate(cases):
        print(f'Case {index + 1}/{len(cases)}: {name}', flush=True)
        table, path = benchmark(contexts, scenario, policies, episodes=episodes, seed=seed, dqn=dqn)
        table['case'], table['purpose'] = name, purpose
        for key, value in scenario.dictionary().items():
            table[key] = value
        tables.append(table)
    folder = save_run('study', {'full_grid': full, 'episodes': episodes,
        'cases': [scenario.dictionary() for _, scenario, _ in cases], 'policies': policies,
        'code_sha256': hashlib.sha256(b''.join(path.read_bytes() for path in sorted((ROOT / 'src').rglob('*.py')))).hexdigest(),
        'dqn_sha256': hashlib.sha256(b''.join(value.detach().numpy().tobytes() for value in dqn.state_dict().values())).hexdigest(),
        'contexts_sha256': hashlib.sha256(contexts['probabilities'].tobytes() + contexts['difficulty'].tobytes()).hexdigest()}, seed)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(folder / 'all_validation_episodes.csv.gz', index=False)
    manifest = pd.DataFrame([{'case': name, **scenario.dictionary()} for name, scenario, _ in full_cases()])
    manifest.to_csv(folder / 'full_grid_manifest.csv', index=False)
    return table, folder


def select_controller(table, full=False):
    relevant = table[(table.model != 'None') & table.policy.isin(['qmdp', 'grid', 'dqn'])]
    if not full:
        relevant = relevant[relevant.purpose == 'core']
    # Average conditions within each matched episode first, preserving pairing across conditions.
    per_episode = relevant.groupby(['policy', 'episode']).reward.mean()
    scores = per_episode.groupby('policy').agg(['mean', 'sem'])
    best = scores['mean'].idxmax()
    eligible = scores.index[scores['mean'] >= scores.loc[best, 'mean'] - scores.loc[best, 'sem']]
    winner = next(name for name in ['qmdp', 'grid', 'dqn'] if name in eligible)
    return winner, scores


def comparison_table(table, winner):
    rows = []
    comparisons = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload',
        'blind', 'myopic', 'rest_threshold', 'qmdp_fixed']
    for case, group in table.groupby('case', sort=False):
        for other in comparisons:
            result = paired_difference(group, winner, other)
            rows.append({'case': case, 'model': group.model.iloc[0], 'beta': group.beta.iloc[0], **result})
    return pd.DataFrame(rows)
