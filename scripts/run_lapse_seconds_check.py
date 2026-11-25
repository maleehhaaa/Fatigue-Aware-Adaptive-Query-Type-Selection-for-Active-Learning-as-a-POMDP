"""Extra check: the Lapse world with the onset counted in seconds instead of answers.

In the main Lapse world ('forced'), each session disengages after U(100, 300) answered questions
(Stage 2; U(20, 100) in Stage 1). A policy that answers faster therefore spends more of the
shift lapsed. The 'forced_seconds' mode (added 21 Nov) times the same lapse in human seconds,
with the range set where always-verify (which ignores lambda) meets the lapse in the answer-count
world: U(75, 310) s per 600-s shift in Stage 2, U(14, 88) s in Stage 1. The lapse still lasts
until the session ends, and the onset draw is shared by all policies. Only the onset changes.

Parts:
  regression  Reruns 8 development-study runs with the current code and checks that they match
              outputs/stage2_study/ea603cfc82 exactly (the answer-count mode must be unchanged).
  stage1      Stage 1, validation people, 500 matched sessions: every answer-count Lapse case and
              its seconds-onset counterpart, all non-DQN policies.
  stage2      Stage 2, development design (validation annotators, seeds 42-51, 50 and 200
              starting labels), 10 policies, in both Lapse worlds with the same code.
  analyse     Tables: policy means, QMDP vs each rival, and the change from answers to seconds.

    python scripts/run_lapse_seconds_check.py regression
    python scripts/run_lapse_seconds_check.py stage1
    python scripts/run_lapse_seconds_check.py stage2 --workers 13
    python scripts/run_lapse_seconds_check.py analyse

This is a development check (report-only). The final test includes the seconds-onset world
through src/eval/stage2.final_test_cases() and src/eval/study.final_test_cases().
Results go to outputs/lapse_seconds_check/.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
from src.common import load_contexts
from src.sim.fatigue import Scenario
import src.eval.study as study
from src.eval.stage1 import benchmark
from src.eval.stats import paired_difference

OUT = project / 'outputs/lapse_seconds_check'
DEVELOPMENT_STUDY = project / 'outputs/stage2_study/ea603cfc82/seed-42'
STAGE1_POLICIES = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload',
                   'blind', 'myopic', 'rest_threshold', 'qmdp', 'qmdp_fixed', 'grid', 'oracle']
STAGE2_POLICIES = ['qmdp', 'qmdp_fixed', 'myopic', 'rest_threshold', 'blind', 'verify', 'two_choice', 'full',
                   'random', 'random_image_full']
READ = dict(keep_default_na=False, na_values=[''])


def stage1_pairs():
    """Each answer-count Lapse case of the initial study, with its seconds-onset counterpart."""
    initial = {name: scenario for name, scenario, _ in study.initial_cases()}
    pairs = []
    for name, scenario, _ in study.lapse_seconds_cases():
        counterpart = name.replace('Lapse_seconds_', 'Lapse_').replace('lapse_seconds_', 'lapse_')
        pairs.append((counterpart, initial[counterpart], name, scenario))
    return pairs


def workers_env():
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'


# ------------------------------------------------------------------ regression
def run_regression():
    import src.eval.stage2 as stage2
    old = pd.read_csv(DEVELOPMENT_STUDY / 'all_runs.csv', **READ)
    config = json.loads((DEVELOPMENT_STUDY / 'config.json').read_text())
    thresholds = dict(zip([s['model'] + str(s['beta']) for s in config['scenarios']], config['rest_thresholds']))
    cases = {(s.model, s.beta): s for s in stage2.experiment_cases()}
    picks = [('Lapse', 1, p, 200, seed) for p in ['qmdp', 'verify', 'blind', 'rest_threshold'] for seed in (42, 43)]
    data, calibration = stage2.load_learning_data(), stage2.load_calibration()
    rows = []
    for model, beta, policy, size, seed in picks:
        options = stage2.default_options()
        if policy == 'rest_threshold':
            options = {**options, 'rest_threshold': thresholds[f'{model}{beta}']}   # same tuned value as the study
        summary, _ = stage2.run_one(cases[(model, float(beta))], policy, size, seed, options, data, calibration)
        before = old[(old.model == model) & (old.beta == beta) & (old.policy == policy) & (old.seed_size == size)
                     & (old.seed == seed)].iloc[0]
        for metric in ['auc_accuracy_percent', 'accuracy_2400s', 'acquired_answers', 'answers_after_lapse']:
            rows.append({'model': model, 'beta': beta, 'policy': policy, 'seed': seed, 'metric': metric,
                         'development_study': before[metric], 'current_code': summary[metric],
                         'identical': bool(np.isclose(before[metric], summary[metric], rtol=0, atol=1e-9))})
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / 'regression_check.csv', index=False)
    print(table.to_string(index=False))
    print('\nAll identical: the answer-count Lapse world is unchanged.' if table.identical.all()
          else '\nNOT identical: stop and check the change before running anything else.')


# ------------------------------------------------------------------ Stage 1
def run_stage1(episodes=500, seed=42):
    OUT.mkdir(parents=True, exist_ok=True)
    contexts = load_contexts()
    tables, paired = [], []
    for count_name, count_scenario, seconds_name, seconds_scenario in stage1_pairs():
        for name, scenario, onset in [(count_name, count_scenario, 'answers'), (seconds_name, seconds_scenario, 'seconds')]:
            table, _ = benchmark(contexts, scenario, STAGE1_POLICIES, episodes=episodes, seed=seed)
            table['case'], table['onset'], table['beta'] = name, onset, scenario.beta
            tables.append(table)
            for rival in [p for p in STAGE1_POLICIES if p not in ('qmdp', 'oracle')]:
                paired.append({'case': name, 'onset': onset, 'beta': scenario.beta,
                               **paired_difference(table, 'qmdp', rival, seed=seed)})
            print(f'{name}: done', flush=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(OUT / 'stage1_episodes.csv.gz', index=False)
    columns = ['reward', 'information', 'seconds', 'error_rate', 'questions'] + [c for c in table if c.startswith('count_')]
    table.groupby(['case', 'onset', 'beta', 'policy'])[columns].mean().round(3).to_csv(OUT / 'stage1_summary.csv')
    pd.DataFrame(paired).round(3).to_csv(OUT / 'stage1_qmdp_vs_rivals.csv', index=False)


# ------------------------------------------------------------------ Stage 2
def run_stage2(workers=6, seeds=range(42, 52), seed_sizes=(50, 200)):
    import src.eval.stage2 as stage2
    OUT.mkdir(parents=True, exist_ok=True)
    count_world = stage2.experiment_cases()[-1]
    seconds_world = stage2.lapse_seconds_case()
    worlds = {'answers': count_world, 'seconds': seconds_world}
    # Greedy-plus-rest threshold tuned per world, with the Stage-1 procedure on training people.
    thresholds = dict(zip(worlds, stage2.tuned_rest_thresholds(list(worlds.values()))))
    print('Greedy-plus-rest thresholds:', thresholds, flush=True)
    options = stage2.default_options()
    jobs = [(onset, (scenario, policy, size, seed,
                     {**options, 'rest_threshold': thresholds[onset]} if policy == 'rest_threshold' else options))
            for onset, scenario in worlds.items() for size in seed_sizes for seed in seeds for policy in STAGE2_POLICIES]
    workers_env()
    print(f'{len(jobs)} classifier runs on {workers} workers...', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=stage2.initialize_worker) as pool:
        futures = {pool.submit(stage2.learning_job, job): onset for onset, job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            summary = future.result()
            summary.update({'onset': futures[future], 'run_folder': Path(summary['run_folder']).as_posix()})
            rows.append(summary)
            if count % 20 == 0 or count == len(jobs):
                print(f'  {count}/{len(jobs)} done', flush=True)
    table = pd.DataFrame(rows).sort_values(['onset', 'seed_size', 'seed', 'policy'])
    table.to_csv(OUT / 'stage2_runs.csv', index=False)
    (OUT / 'stage2_settings.json').write_text(json.dumps({'worlds': {k: v.dictionary() for k, v in worlds.items()},
        'rest_thresholds': thresholds, 'seeds': list(seeds), 'seed_sizes': list(seed_sizes),
        'annotators': 'validation (development)'}, indent=2))


def bootstrap(values, seed=42, resamples=4000):
    values = np.asarray(values, float)
    means = values[np.random.default_rng(seed).integers(len(values), size=(resamples, len(values)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return values.mean(), low, high


def analyse():
    outputs = {}
    if (OUT / 'stage2_runs.csv').exists():
        table = pd.read_csv(OUT / 'stage2_runs.csv', **READ)
        metrics = ['auc_accuracy_percent', 'accuracy_2400s', 'acquired_answers', 'acquired_error_rate', 'answers_after_lapse']
        summary = table.groupby(['onset', 'seed_size', 'policy'])[metrics].mean()
        summary['share_answers_after_lapse'] = summary.answers_after_lapse / summary.acquired_answers
        summary.round(3).to_csv(OUT / 'stage2_summary.csv')
        paired, change = [], []
        for (onset, size), part in table.groupby(['onset', 'seed_size']):
            wide = part.pivot(index='seed', columns='policy', values='auc_accuracy_percent')
            for rival in [p for p in STAGE2_POLICIES if p != 'qmdp']:
                mean, low, high = bootstrap(wide['qmdp'] - wide[rival])
                paired.append({'onset': onset, 'seed_size': size, 'rival': rival, 'qmdp_minus_rival': mean,
                               'ci_low': low, 'ci_high': high})
            by_policy = part.groupby('policy')[['auc_accuracy_percent', 'answers_after_lapse']].mean()
            ranks = by_policy.rank()
            outputs[f'rank correlation, average accuracy vs answers after the lapse ({onset}, {size} labels)'] = \
                float(np.corrcoef(ranks.auc_accuracy_percent, ranks.answers_after_lapse)[0, 1])
        for (size, policy), part in table.groupby(['seed_size', 'policy']):
            wide = part.pivot(index='seed', columns='onset', values='auc_accuracy_percent')
            mean, low, high = bootstrap(wide['seconds'] - wide['answers'])
            change.append({'seed_size': size, 'policy': policy, 'seconds_minus_answers': mean, 'ci_low': low, 'ci_high': high})
        paired = pd.DataFrame(paired).round(3)
        paired.to_csv(OUT / 'stage2_qmdp_vs_rivals.csv', index=False)
        pd.DataFrame(change).round(3).to_csv(OUT / 'stage2_change_answers_to_seconds.csv', index=False)
        best = paired.loc[paired.groupby(['onset', 'seed_size']).qmdp_minus_rival.idxmin()]
        best.to_csv(OUT / 'stage2_qmdp_vs_strongest.csv', index=False)
        print(summary.round(2).to_string(), '\n')
        print('QMDP minus the strongest rival:\n', best.to_string(index=False), '\n')
    if (OUT / 'stage1_qmdp_vs_rivals.csv').exists():
        stage1_table = pd.read_csv(OUT / 'stage1_qmdp_vs_rivals.csv', **READ)
        print('Stage 1, QMDP minus verify and greedy-plus-rest:\n',
              stage1_table[stage1_table.second.isin(['verify', 'rest_threshold', 'myopic'])].to_string(index=False))
    (OUT / 'analysis_notes.json').write_text(json.dumps(outputs, indent=2))
    for key, value in outputs.items():
        print(f'{key}: {value:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['regression', 'stage1', 'stage2', 'analyse'])
    parser.add_argument('--workers', type=int, default=6)
    arguments = parser.parse_args()
    {'regression': run_regression, 'stage1': run_stage1, 'stage2': lambda: run_stage2(arguments.workers),
     'analyse': analyse}[arguments.part]()
