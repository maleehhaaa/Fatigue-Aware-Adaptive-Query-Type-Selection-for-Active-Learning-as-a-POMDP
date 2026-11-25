"""E1b: does the Lapse result depend on lambda when the lapse is timed in seconds?

The lambda sweep (scripts/run_lambda_sweep.py, 21 Nov) varied lambda only in the Lapse world whose
onset is counted in answers ('forced'). This check repeats that design in BOTH Lapse worlds with the
frozen code, so the two can be compared side by side:
  - 'answers' world: Lapse beta 1, onset after U(100, 300) answered questions (as in the sweep).
  - 'seconds' world: Lapse beta 1, onset after U(75, 310) human seconds per 600-s shift
    (src/eval/stage2.lapse_seconds_case()).
Design copied from the lambda sweep: TRAINING annotators, tuning seeds 100-109, 200 starting labels,
lambda grid 0.25, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25. The lambda-aware policies (QMDP, greedy,
greedy-plus-rest, fatigue-blind) run at every lambda; always-verify and always-2-choice ignore lambda
and run once per world. Greedy-plus-rest is re-tuned for every (world, lambda) on training people
with the Stage-1 procedure (src/eval/stage2.tuned_rest_thresholds).

This is a development sensitivity check. It never touches the test annotators or the final-test
images, and it cannot change lambda: lambda = 1.0 stays the project's choice whatever the outcome.

Parts:
  smoke     Tiny version of every part, separate folder. Not a result.
  stage1    Writes the predictions (once, before any run), tunes greedy-plus-rest for every
            (world, lambda), and records QMDP's Stage-1 information rate and pauses on training people.
            As in the lambda sweep, Stage 1 here uses the Stage-2 onsets, so in the answers world the
            lapse rarely starts within a 150-step session: these rates describe speed and pausing only.
  stage2    The 600 classifier runs (2 worlds x (4 policies x 7 lambdas + 2 fixed formats) x 10 seeds).
  analyse   Tables, QMDP against each rival at the same lambda, the prediction checks and a figure.

    python scripts/run_lapse_lambda_check.py smoke
    python scripts/run_lapse_lambda_check.py stage1
    python scripts/run_lapse_lambda_check.py stage2 --workers 13
    python scripts/run_lapse_lambda_check.py analyse

The predictions are written to outputs/lapse_lambda_check/predictions.json before any run and are
never edited afterwards. src/ is not changed. Results go to outputs/lapse_lambda_check/.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
from src.common import load_calibration, load_contexts
from src.control.qmdp import PlanningModel
import src.eval.stage1 as stage1
import src.eval.stage2 as stage2

OUT = project / 'outputs/lapse_lambda_check'
READ = dict(keep_default_na=False, na_values=[''])

# ---- Settings fixed before running (same as the lambda sweep; change only here, never after the runs) ----
WORLDS = {'answers': stage2.experiment_cases()[4],      # Lapse beta 1, onset U(100, 300) answers
          'seconds': stage2.lapse_seconds_case()}       # Lapse beta 1, onset U(75, 310) seconds
GRID = (.25, .5, .625, .75, .875, 1., 1.25)
LAMBDA_AWARE = ('qmdp', 'myopic', 'rest_threshold', 'blind')
REFERENCES = ('verify', 'two_choice')
SEEDS = tuple(range(100, 110))
SEED_SIZE = 200
PRIMARY = 'auc_accuracy_percent'
ANNOTATORS = 'train (E1b lapse lambda check, tuning only)'

PREDICTIONS = {
    'basis': ('Written before any E1b run, from the lambda sweep (outputs/lambda_sweep: answers world, '
              'training annotators, seeds 100-109) and the seconds-onset check at lambda = 1 '
              '(outputs/lapse_seconds_check, final test).'),
    'decision': 'Lambda stays 1.0 whatever the outcome; this is a sensitivity check only.',
    'P1': ('Reproduction, answers world: QMDP average accuracy is higher at lambda 0.25 than at 1.25 '
           '(sweep: 74.9 vs 69.3), and QMDP minus always-2-choice is not clearly negative at 0.25 '
           '(sweep: about +1.0) but clearly negative at 1.0 (sweep: -3.9).'),
    'P2': ('Mechanism, both worlds: QMDP collects more answers, and more answers after the lapse, '
           'at lambda 1.25 than at 0.25.'),
    'P3': ('Main: QMDP average accuracy falls less from lambda 0.25 to 1.25 in the seconds world than '
           'in the answers world (timing the lapse in seconds removes part of the throughput penalty).'),
    'P4': ('Main: in the seconds world, QMDP minus its strongest rival (the best of greedy, '
           'greedy-plus-rest and fatigue-blind at the same lambda, always-verify and always-2-choice) '
           'is higher at lambda 0.25 than at lambda 1.0, i.e. the remaining Lapse loss is partly '
           'driven by lambda.'),
    'P5': ('Reproduction on other people: at lambda 1.0 in the seconds world, QMDP is behind its '
           'strongest rival (development check -1.70, final test -1.84 at 200 labels).')}


def code_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted((project / 'src').rglob('*.py')))).hexdigest()


def check_frozen():
    frozen = json.loads((project / 'configs/frozen_src.json').read_text())['src_sha256']
    if code_hash() != frozen:
        raise SystemExit('src/ differs from configs/frozen_src.json. Stop and check before running.')
    return frozen


def write_predictions():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'predictions.json'
    if not path.exists():
        path.write_text(json.dumps({**PREDICTIONS, 'written_utc': datetime.now(timezone.utc).isoformat(),
                                    'worlds': {k: v.dictionary() for k, v in WORLDS.items()}, 'grid': GRID,
                                    'seeds': SEEDS, 'seed_size': SEED_SIZE, 'src_sha256': check_frozen()}, indent=2))
        print('Predictions written to', path)


def workers_env():
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'


# ------------------------------------------------------------------ Stage 1
def run_stage1(grid=None, worlds=None, episodes=300):
    check_frozen()
    write_predictions()
    grid, worlds = grid or GRID, worlds or WORLDS
    contexts, calibration = load_contexts(), load_calibration()
    population = stage1.population_model(calibration)
    keys = [(name, lam) for name in worlds for lam in grid]
    thresholds = stage2.tuned_rest_thresholds([replace(worlds[name], lam=lam) for name, lam in keys])
    threshold_table = pd.DataFrame([{'world': n, 'lam': l, 'rest_threshold': t} for (n, l), t in zip(keys, thresholds)])
    threshold_table.to_csv(OUT / 'rest_thresholds.csv', index=False)
    rows = []
    for (name, lam), threshold in zip(keys, thresholds):
        world = replace(worlds[name], lam=lam)
        model = PlanningModel(contexts, world, population)
        model.rest_threshold = threshold
        for policy in ('qmdp', 'myopic', 'rest_threshold', 'verify'):
            result = stage1.run_policy(contexts, world, policy, 'train', episodes, 542, model, calibration=calibration)
            information, seconds = result['information'].mean(), result['seconds'].mean()
            rows.append({'world': name, 'lam': lam, 'policy': policy, 'information': information, 'seconds': seconds,
                         'bits_per_second': information / seconds, 'pauses': result['action_counts'][:, 0].mean()})
        print(f'Stage 1 {name} lambda {lam}: threshold {threshold}', flush=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / 'stage1_rates.csv', index=False)


# ------------------------------------------------------------------ Stage 2
def initialize_train_worker():
    """Same worker as the study, but the simulated people come from the TRAINING split."""
    stage2.initialize_worker()
    stage2.make_session_factory = partial(stage2.make_session_factory, split='train')


def run_stage2(workers=6, grid=None, worlds=None, seeds=SEEDS, options=None, tag='stage2_runs'):
    check_frozen()
    write_predictions()
    if not (OUT / 'rest_thresholds.csv').exists():
        raise SystemExit('Run the stage1 part first: it tunes greedy-plus-rest for every (world, lambda).')
    grid, worlds = grid or GRID, worlds or WORLDS
    thresholds = pd.read_csv(OUT / 'rest_thresholds.csv', **READ).set_index(['world', 'lam']).rest_threshold
    options = {**(options or stage2.default_options()), 'annotators': ANNOTATORS,
               'evaluation': 'development validation images; training annotators (E1b)'}
    jobs = []
    for name, scenario in worlds.items():
        for seed in seeds:
            for policy in REFERENCES:
                jobs.append(((name, None), (scenario, policy, SEED_SIZE, seed, options)))
            for lam in grid:
                for policy in LAMBDA_AWARE:
                    job_options = {**options, 'rest_threshold': float(thresholds[(name, lam)])} if policy == 'rest_threshold' else options
                    jobs.append(((name, lam), (replace(scenario, lam=lam), policy, SEED_SIZE, seed, job_options)))
    workers_env()
    print(f'{len(jobs)} classifier runs on {workers} workers...', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_train_worker) as pool:
        futures = {pool.submit(stage2.learning_job, job): key for key, job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            summary = future.result()
            name, lam = futures[future]
            summary.update({'world': name, 'lam': lam, 'run_folder': Path(summary['run_folder']).as_posix()})
            rows.append(summary)
            if count % 50 == 0 or count == len(jobs):
                print(f'  {count}/{len(jobs)} done', flush=True)
    table = pd.DataFrame(rows).sort_values(['world', 'policy', 'lam', 'seed'])
    assert not table.final_test_evaluated.astype(str).eq('True').any(), 'final-test images must not be used'
    table.to_csv(OUT / f'{tag}.csv', index=False)
    (OUT / f'{tag}_settings.json').write_text(json.dumps({
        'worlds': {k: v.dictionary() for k, v in worlds.items()}, 'grid': list(grid), 'seeds': list(seeds),
        'seed_size': SEED_SIZE, 'annotators': ANNOTATORS, 'src_sha256': code_hash(),
        'finished_utc': datetime.now(timezone.utc).isoformat()}, indent=2))


# ------------------------------------------------------------------ analysis
def bootstrap(values, seed=42, resamples=4000):
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    means = values[np.random.default_rng(seed).integers(len(values), size=(resamples, len(values)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return float(values.mean()), float(low), float(high)


def per_seed(table, world, policy, lam=None):
    part = table[(table.world == world) & (table.policy == policy)]
    if lam is not None:
        part = part[np.isclose(part.lam, lam)]
    return part.set_index('seed')[PRIMARY]


def analyse(tag='stage2_runs'):
    table = pd.read_csv(OUT / f'{tag}.csv', **READ)
    table['lam'] = pd.to_numeric(table.lam, errors='coerce')
    grid = sorted(table.lam.dropna().unique())
    metrics = [c for c in [PRIMARY, 'accuracy_2400s', 'acquired_answers', 'acquired_error_rate', 'answers_after_lapse']
               if c in table]
    summary = table.groupby(['world', 'policy', 'lam'], dropna=False)[metrics].mean().round(3)
    summary.to_csv(OUT / 'summary_by_world_policy_lambda.csv')

    comparisons, strongest = [], []
    for world in table.world.unique():
        for lam in grid:
            qmdp = per_seed(table, world, 'qmdp', lam)
            rivals = {p: per_seed(table, world, p, lam) for p in LAMBDA_AWARE if p != 'qmdp'}
            rivals.update({p: per_seed(table, world, p) for p in REFERENCES})
            rows = []
            for rival, values in rivals.items():
                mean, low, high = bootstrap((qmdp - values).dropna())
                rows.append({'world': world, 'lam': lam, 'rival': rival, 'qmdp_minus_rival': mean, 'ci_low': low, 'ci_high': high})
            comparisons += rows
            strongest.append(min(rows, key=lambda r: r['qmdp_minus_rival']))
    comparisons, strongest = pd.DataFrame(comparisons).round(3), pd.DataFrame(strongest).round(3)
    comparisons.to_csv(OUT / 'qmdp_vs_rivals_same_lambda.csv', index=False)
    strongest.to_csv(OUT / 'qmdp_vs_strongest_rival.csv', index=False)
    print(strongest.to_string(index=False))
    checks = check_predictions(table, summary, comparisons, strongest, grid)
    plot(table, strongest, grid)
    return checks


def check_predictions(table, summary, comparisons, strongest, grid):
    low, high, one = min(grid), max(grid), 1.0
    have = lambda *values: all(any(np.isclose(v, g) for g in grid) for v in values)
    worlds = list(table.world.unique())

    def qmdp_mean(world, lam):
        return float(summary.loc[(world, 'qmdp', lam), PRIMARY])

    def compare(world, lam, rival):
        row = comparisons[(comparisons.world == world) & np.isclose(comparisons.lam, lam) & (comparisons.rival == rival)]
        return row.iloc[0].to_dict() if len(row) else None

    results = {}
    if 'answers' in worlds and have(low, high, one):
        at_low, at_one = compare('answers', low, 'two_choice'), compare('answers', one, 'two_choice')
        results['P1'] = {'qmdp_auc_low_high': [qmdp_mean('answers', low), qmdp_mean('answers', high)],
                         'qmdp_minus_two_choice_low': at_low, 'qmdp_minus_two_choice_1.0': at_one,
                         'holds': bool(qmdp_mean('answers', low) > qmdp_mean('answers', high)
                                       and at_low['ci_high'] >= 0 and at_one['ci_high'] < 0)}
    else:
        results['P1'] = {'holds': None}
    if have(low, high):
        p2 = {}
        for world in worlds:
            answers = [float(summary.loc[(world, 'qmdp', l), 'acquired_answers']) for l in (low, high)]
            after = [float(summary.loc[(world, 'qmdp', l), 'answers_after_lapse']) for l in (low, high)]
            p2[world] = {'answers_low_high': answers, 'after_lapse_low_high': after,
                         'holds': bool(answers[1] > answers[0] and after[1] > after[0])}
        results['P2'] = {**p2, 'holds': all(v['holds'] for v in p2.values())}
        if {'answers', 'seconds'} <= set(worlds):
            drops = {w: per_seed(table, w, 'qmdp', low) - per_seed(table, w, 'qmdp', high) for w in ('answers', 'seconds')}
            mean, lo, hi = bootstrap((drops['answers'] - drops['seconds']).dropna())
            results['P3'] = {'drop_answers': float(drops['answers'].mean()), 'drop_seconds': float(drops['seconds'].mean()),
                             'answers_minus_seconds': mean, 'ci_low': lo, 'ci_high': hi,
                             'holds': bool(drops['seconds'].mean() < drops['answers'].mean())}
        else:
            results['P3'] = {'holds': None}
    else:
        results['P2'] = results['P3'] = {'holds': None}
    if 'seconds' in worlds and have(low, one):
        gap = strongest[strongest.world == 'seconds'].set_index('lam')
        results['P4'] = {'strongest_at_low': gap.loc[low].to_dict(), 'strongest_at_1.0': gap.loc[one].to_dict(),
                         'holds': bool(gap.loc[low, 'qmdp_minus_rival'] > gap.loc[one, 'qmdp_minus_rival'])}
        results['P5'] = {'strongest_at_1.0': gap.loc[one].to_dict(), 'holds': bool(gap.loc[one, 'qmdp_minus_rival'] < 0)}
    else:
        results['P4'] = results['P5'] = {'holds': None}
    (OUT / 'prediction_checks.json').write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: v.get('holds') for k, v in results.items()}, indent=2))
    return results


def plot(table, strongest, grid):
    import importlib.util
    spec = importlib.util.spec_from_file_location('final_figures', project / 'scripts' / 'make_final_test_figures.py')
    figs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs)
    plt = figs.plt
    worlds = [w for w in ('answers', 'seconds') if w in table.world.unique()]
    fig, axes = plt.subplots(1, len(worlds) + 1, figsize=(4.2 * (len(worlds) + 1), 3.8))
    for ax, world in zip(axes, worlds):
        part = table[table.world == world]
        for policy in LAMBDA_AWARE:
            means = part[part.policy == policy].groupby('lam')[PRIMARY].agg(['mean', 'sem'])
            ax.errorbar(means.index, means['mean'], yerr=1.96 * means['sem'], color=figs.COLOR[policy],
                        marker=figs.MARKER[policy], linestyle=figs.DASH[policy], linewidth=2.2 if policy == 'qmdp' else 1.5,
                        markersize=5, capsize=0, label=figs.NAMES[policy])
        for policy in REFERENCES:
            value = part[part.policy == policy][PRIMARY].mean()
            ax.axhline(value, color=figs.COLOR[policy], linestyle=figs.DASH[policy], linewidth=1.2, label=figs.NAMES[policy])
        ax.axvline(1.0, color=figs.MUTED, linewidth=1, linestyle=':')
        ax.set(xlabel='λ (bits per human second)', title=f'Lapse, onset in {world}')
    axes[0].set_ylabel('Average accuracy 0–2,400 s (%)')
    ax = axes[-1]
    for world, style in zip(worlds, ['-', '--']):
        gap = strongest[strongest.world == world].sort_values('lam')
        ax.plot(gap.lam, gap.qmdp_minus_rival, color=figs.COLOR['qmdp'], linestyle=style, marker='o', label=f'onset in {world}')
        ax.fill_between(gap.lam, gap.ci_low, gap.ci_high, color=figs.COLOR['qmdp'], alpha=.12, linewidth=0)
    ax.axhline(0, color=figs.INK2, linewidth=1)
    ax.axvline(1.0, color=figs.MUTED, linewidth=1, linestyle=':')
    ax.set(xlabel='λ (bits per human second)', ylabel='QMDP minus strongest rival (points)',
           title='QMDP against its strongest rival')
    ax.legend(frameon=False, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, -.04))
    fig.suptitle('E1b (development check, training annotators, 200 starting labels, 10 seeds): λ in both Lapse worlds',
                 x=.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(OUT / 'lapse_lambda_check.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote lapse_lambda_check.png')


# ------------------------------------------------------------------ smoke
def smoke():
    """Tiny version of every part, separate folder. Not a result."""
    global OUT
    OUT = project / 'outputs/lapse_lambda_check_smoke'
    run_stage1(grid=(.25, 1.), episodes=40)
    options = {**stage2.default_options(), 'budgets_seconds': [150, 300]}
    run_stage2(workers=2, grid=(.25, 1.), seeds=(100,), options=options)
    analyse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['smoke', 'stage1', 'stage2', 'analyse'])
    parser.add_argument('--workers', type=int, default=6)
    arguments = parser.parse_args()
    {'smoke': smoke, 'stage1': run_stage1, 'stage2': lambda: run_stage2(arguments.workers),
     'analyse': analyse}[arguments.part]()
