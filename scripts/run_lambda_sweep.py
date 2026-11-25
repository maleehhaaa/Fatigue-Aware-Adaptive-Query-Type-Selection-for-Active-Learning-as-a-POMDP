"""Lambda sweep: choose the price of one human second (bits per second) by a rule fixed in advance.

Two parts:

  theory  Stage 1, training people only. For each tuning world, trace QMDP's information rate
          (bits per human second) against lambda and find the Dinkelbach fixed point
          lambda = information / seconds. Under a hard time budget, that rate is the
          opportunity cost of one second, so the fixed point is the theory value. This part
          also tunes greedy-plus-rest's pause threshold for every (lambda, world) pair, with the
          same Stage-1 procedure used in Notebook 08.
  stage2  Classifier learning with TRAINING annotators and tuning seeds 100-109, so nothing
          overlaps the development study (validation annotators, seeds 42-51) or the final test.
          Every lambda-aware policy runs at every lambda; fixed-format references run once.
  analyse Applies the selection rule written by 'theory' before any Stage-2 run, and saves
          the choice, tables and plots.

    python scripts/run_lambda_sweep.py theory
    python scripts/run_lambda_sweep.py stage2 --workers 10
    python scripts/run_lambda_sweep.py analyse
    python scripts/run_lambda_sweep.py smoke          # 2-minute check that everything runs

src/ is not changed. Results go to outputs/lambda_sweep/.
"""
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
from src.common import load_calibration, load_contexts
from src.control.qmdp import PlanningModel
from src.sim.fatigue import Scenario
import src.eval.stage1 as stage1
import src.eval.stage2 as stage2

OUT = project / 'outputs/lambda_sweep'

# ---- Settings fixed before running (change them here, never after seeing Stage-2 results) ----
SELECTION_WORLDS = {            # used to choose lambda, equal weight
    'None': Scenario(model='None', beta=0),
    'Markov1': Scenario(beta=1),
    'Markov2': Scenario(beta=2),
    'FALCON1': Scenario(model='FALCON', beta=1)}
REPORT_ONLY_WORLDS = {          # reported, never used for the choice
    'Lapse1': Scenario(model='Lapse', beta=1, lapse_onset_min=100., lapse_onset_max=300.)}
THEORY_GRID = tuple(np.round(np.arange(.1, 1.51, .1), 2))
STAGE2_GRID = (.25, .5, .625, .75, .875, 1., 1.25)
LAMBDA_AWARE = ('qmdp', 'myopic', 'rest_threshold', 'blind')   # read lambda from the scenario
REFERENCES = ('verify', 'two_choice')                          # ignore lambda: run once
TUNE_SEEDS = tuple(range(100, 110))                            # disjoint from 42-51
SEED_SIZE = 200
PRIMARY = 'auc_accuracy_percent'                               # average accuracy over 0-2,400 s
RULE = ('Primary metric: QMDP average accuracy over 0-2400 s, 200 starting labels, training '
        'annotators, tuning seeds 100-109, averaged with equal weight over None, Markov beta=1, '
        'Markov beta=2 and FALCON beta=1. Lapse is reported but not used. Step 1: for each lambda, '
        'the paired difference to the best lambda (per seed, averaged over the four worlds) gets a '
        'bootstrap 95% interval (4,000 resamples, seed 42). Step 2: candidates are all lambdas whose '
        'interval reaches 0 (not clearly worse than the best). Step 3: among candidates, choose the '
        'one closest to the Stage-1 theory value (median Dinkelbach fixed point over the four worlds); '
        'ties go to the smaller lambda. The same lambda is then used by every lambda-aware policy.')


def code_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted((project / 'src').rglob('*.py')))).hexdigest()


def environment():
    import torch
    return {'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'torch': torch.__version__, 'src_sha256': code_hash()}


# ------------------------------------------------------------------ theory (Stage 1)
def information_rate(contexts, calibration, population, scenario, lam, episodes=300, seed=542):
    """QMDP's bits per second on training people at this lambda."""
    world = replace(scenario, lam=float(lam))
    model = PlanningModel(contexts, world, population)
    result = stage1.run_policy(contexts, world, 'qmdp', 'train', episodes, seed, model, calibration=calibration)
    information, seconds = result['information'].mean(), result['seconds'].mean()
    return information / seconds, information, seconds


def dinkelbach(contexts, calibration, population, scenario, start=.25, tolerance=.005, steps=12):
    """Iterate lambda <- information / seconds until it stops moving."""
    path, lam = [start], start
    for _ in range(steps):
        new = information_rate(contexts, calibration, population, scenario, lam)[0]
        path.append(float(new))
        if abs(new - lam) < tolerance:
            break
        lam = new
    return path


def run_theory():
    OUT.mkdir(parents=True, exist_ok=True)
    contexts, calibration = load_contexts(), load_calibration()
    population = stage1.population_model(calibration)
    rows, fixed_points, thresholds = [], {}, []
    for name, scenario in {**SELECTION_WORLDS, **REPORT_ONLY_WORLDS}.items():
        for lam in THEORY_GRID:
            rate, information, seconds = information_rate(contexts, calibration, population, scenario, lam)
            rows.append({'world': name, 'lam': lam, 'bits_per_second': rate, 'information': information,
                         'seconds': seconds, 'stage1_reward': information - lam * seconds})
        path = dinkelbach(contexts, calibration, population, scenario)
        fixed_points[name] = {'fixed_point': path[-1], 'path': path}
        print(f'{name:8s} Dinkelbach path: ' + ' -> '.join(f'{x:.3f}' for x in path), flush=True)
        for lam in STAGE2_GRID:
            world = replace(scenario, lam=lam)
            model = PlanningModel(contexts, world, population)
            best = stage1.tune_rest_threshold(contexts, world, model, calibration, seed=42)
            thresholds.append({'world': name, 'lam': lam, 'rest_threshold': best})
    pd.DataFrame(rows).to_csv(OUT / 'stage1_rate_curve.csv', index=False)
    pd.DataFrame(thresholds).to_csv(OUT / 'rest_thresholds.csv', index=False)
    theory = float(np.median([fixed_points[name]['fixed_point'] for name in SELECTION_WORLDS]))
    rule = {'written_utc': datetime.now(timezone.utc).isoformat(), 'rule': RULE,
            'theory_lambda': theory, 'fixed_points': fixed_points, 'stage2_grid': STAGE2_GRID,
            'tune_seeds': TUNE_SEEDS, 'selection_worlds': list(SELECTION_WORLDS),
            'report_only_worlds': list(REPORT_ONLY_WORLDS), 'environment': environment()}
    (OUT / 'selection_rule.json').write_text(json.dumps(rule, indent=2))
    print(f'Theory value (median fixed point): {theory:.3f}. Rule saved before any Stage-2 run.')


# ------------------------------------------------------------------ Stage 2
def initialize_train_worker():
    """Same worker as the study, but the simulated people come from the TRAINING split."""
    stage2.initialize_worker()
    stage2.make_session_factory = partial(stage2.make_session_factory, split='train')


def stage2_jobs(lambdas, worlds, seeds, options):
    thresholds = pd.read_csv(OUT / 'rest_thresholds.csv', keep_default_na=False, na_values=['']).set_index(['world', 'lam']).rest_threshold
    jobs = []
    for name, scenario in worlds.items():
        for seed in seeds:
            for policy in REFERENCES:
                jobs.append((name, None, (scenario, policy, SEED_SIZE, seed, options)))
            for lam in lambdas:
                for policy in LAMBDA_AWARE:
                    job_options = dict(options)
                    if policy == 'rest_threshold':
                        job_options['rest_threshold'] = float(thresholds[(name, lam)])
                    jobs.append((name, lam, (replace(scenario, lam=lam), policy, SEED_SIZE, seed, job_options)))
    return jobs


def run_stage2(workers, lambdas=None, worlds=None, seeds=TUNE_SEEDS, options=None, tag='runs'):
    if not (OUT / 'selection_rule.json').exists():
        raise SystemExit('Run the theory part first: it writes the selection rule before any Stage-2 run.')
    worlds = worlds or {**SELECTION_WORLDS, **REPORT_ONLY_WORLDS}
    lambdas = lambdas or STAGE2_GRID
    options = options or stage2.default_options()
    # Separate cache key from the development study, and say which people were used.
    options = {**options, 'annotators': 'train (lambda sweep tuning only)'}
    jobs = stage2_jobs(lambdas, worlds, seeds, options)
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    print(f'{len(jobs)} classifier runs on {workers} workers...', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_train_worker) as pool:
        futures = {pool.submit(stage2.learning_job, job): (name, lam) for name, lam, job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            name, lam = futures[future]
            summary = future.result()
            summary.update({'world': name, 'lam': lam, 'run_folder': Path(summary['run_folder']).as_posix()})
            rows.append(summary)
            if count % 50 == 0 or count == len(jobs):
                print(f'  {count}/{len(jobs)} done', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f'{tag}.csv', index=False)
    (OUT / f'{tag}_environment.json').write_text(json.dumps(environment(), indent=2))
    return table


# ------------------------------------------------------------------ analysis
def bootstrap(values, seed=42, resamples=4000):
    random = np.random.default_rng(seed)
    means = values[random.integers(len(values), size=(resamples, len(values)))].mean(1)
    return np.quantile(means, [.025, .975])


def analyse(tag='runs'):
    rule = json.loads((OUT / 'selection_rule.json').read_text())
    # 'None' is a world name, not a missing value.
    table = pd.read_csv(OUT / f'{tag}.csv', keep_default_na=False, na_values=[''])
    table['lam'] = pd.to_numeric(table.lam, errors='coerce')   # empty for the fixed formats
    adaptive = table[table.policy.isin(LAMBDA_AWARE)].copy()
    columns = [c for c in [PRIMARY, 'accuracy_2400s', 'acquired_answers', 'acquired_error_rate',
                           'pool_exhausted', 'answers_after_lapse'] if c in adaptive]
    summary = adaptive.groupby(['world', 'policy', 'lam'])[columns].mean().round(3)
    summary.to_csv(OUT / 'summary_by_world_policy_lambda.csv')

    def choose(policy):
        rows = adaptive[(adaptive.policy == policy) & adaptive.world.isin(rule['selection_worlds'])]
        # One number per seed and lambda: the mean over the selection worlds (equal weight).
        per_seed = rows.pivot_table(index='seed', columns='lam', values=PRIMARY, aggfunc='mean')
        complete = rows.groupby(['seed', 'lam']).world.nunique().unstack() == len(rule['selection_worlds'])
        per_seed = per_seed.where(complete)
        best = per_seed.mean().idxmax()
        out = []
        for lam in per_seed.columns:
            difference = (per_seed[lam] - per_seed[best]).dropna().to_numpy()
            low, high = bootstrap(difference) if lam != best else (0., 0.)
            out.append({'policy': policy, 'lam': lam, 'mean_primary': per_seed[lam].mean(),
                        'minus_best': difference.mean(), 'ci_low': low, 'ci_high': high,
                        'candidate': bool(high >= 0)})
        out = pd.DataFrame(out)
        candidates = out[out.candidate].copy()
        candidates['distance'] = (candidates.lam - rule['theory_lambda']).abs()
        chosen = float(candidates.sort_values(['distance', 'lam']).lam.iloc[0])
        return out, float(best), chosen

    decisions, tables = {}, []
    for policy in LAMBDA_AWARE:
        out, best, chosen = choose(policy)
        tables.append(out)
        decisions[policy] = {'best_mean_lambda': best, 'rule_choice': chosen,
                             'candidates': out[out.candidate].lam.tolist()}
    pd.concat(tables).round(3).to_csv(OUT / 'selection_table.csv', index=False)
    chosen = decisions['qmdp']['rule_choice']

    # Fairness check: QMDP at the chosen lambda vs each baseline at the same lambda and at its own best.
    fairness = []
    for world, group in table.groupby('world'):
        qmdp = group[(group.policy == 'qmdp') & np.isclose(group.lam, chosen)].set_index('seed')[PRIMARY]
        for policy in ['myopic', 'rest_threshold', 'blind']:
            for label, lam in [('same lambda', chosen), ('own best lambda', decisions[policy]['rule_choice'])]:
                rival = group[(group.policy == policy) & np.isclose(group.lam, lam)].set_index('seed')[PRIMARY]
                difference = (qmdp - rival).dropna().to_numpy()
                low, high = bootstrap(difference)
                fairness.append({'world': world, 'rival': policy, 'rival_lambda': lam, 'setting': label,
                                 'qmdp_minus_rival': difference.mean(), 'ci_low': low, 'ci_high': high})
        for policy in REFERENCES:
            rival = group[group.policy == policy].set_index('seed')[PRIMARY]
            difference = (qmdp - rival).dropna().to_numpy()
            low, high = bootstrap(difference)
            fairness.append({'world': world, 'rival': policy, 'rival_lambda': None, 'setting': 'fixed format',
                             'qmdp_minus_rival': difference.mean(), 'ci_low': low, 'ci_high': high})
    pd.DataFrame(fairness).round(3).to_csv(OUT / 'qmdp_vs_baselines_at_choice.csv', index=False)

    decision = {'chosen_lambda': chosen, 'theory_lambda': rule['theory_lambda'], 'per_policy': decisions,
                'rule': rule['rule'], 'rule_written_utc': rule['written_utc'],
                'analysed_utc': datetime.now(timezone.utc).isoformat()}
    (OUT / 'decision.json').write_text(json.dumps(decision, indent=2))
    plot(adaptive, rule, chosen)
    print(json.dumps(decision, indent=2))


def plot(adaptive, rule, chosen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    worlds = sorted(adaptive.world.unique())
    fig, axes = plt.subplots(1, len(worlds) + 1, figsize=(4 * (len(worlds) + 1), 3.6))
    for ax, world in zip(axes, worlds):
        for policy, group in adaptive[adaptive.world == world].groupby('policy'):
            means = group.groupby('lam')[PRIMARY].agg(['mean', 'sem'])
            ax.errorbar(means.index, means['mean'], yerr=means['sem'], marker='o', capsize=2, label=policy)
        ax.axvline(chosen, color='grey', ls='--', lw=1)
        ax.set(title=world, xlabel='lambda (bits per second)', ylabel='average accuracy, 0-2400 s (%)')
    curve = pd.read_csv(OUT / 'stage1_rate_curve.csv', keep_default_na=False, na_values=[''])
    for world, group in curve.groupby('world'):
        axes[-1].plot(group.lam, group.bits_per_second, marker='.', label=world)
    axes[-1].plot([0, 1.5], [0, 1.5], color='black', lw=.8, label='rate = lambda')
    axes[-1].axvline(rule['theory_lambda'], color='grey', ls=':', lw=1)
    axes[-1].set(title='Stage 1: QMDP information rate', xlabel='lambda', ylabel='bits per second')
    axes[0].legend(fontsize=7)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'lambda_sweep.png', dpi=150)


# ------------------------------------------------------------------ smoke test
def smoke():
    """Tiny version of every step, written to a separate folder. Not a result."""
    global OUT, THEORY_GRID, STAGE2_GRID, SELECTION_WORLDS, REPORT_ONLY_WORLDS
    OUT = project / 'outputs/lambda_sweep_smoke'
    THEORY_GRID, STAGE2_GRID = (.25, .75), (.25, .75)
    SELECTION_WORLDS = {'Markov1': Scenario(beta=1)}
    REPORT_ONLY_WORLDS = {}
    run_theory()
    options = {**stage2.default_options(), 'budgets_seconds': [150, 300]}
    run_stage2(workers=2, seeds=(100, 101), options=options)
    analyse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['theory', 'stage2', 'analyse', 'smoke'])
    parser.add_argument('--workers', type=int, default=6)
    arguments = parser.parse_args()
    if arguments.part == 'theory':
        run_theory()
    elif arguments.part == 'stage2':
        run_stage2(arguments.workers)
    elif arguments.part == 'analyse':
        analyse()
    else:
        smoke()
