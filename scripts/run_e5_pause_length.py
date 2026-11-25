"""E5: how much do the results depend on the 10-s pause?

Pause lengths 5, 10 (reference) and 20 s, under two recovery rules:
  A  recovery per pause fixed at 0.35 (the current model): only the price of a pause changes.
  B  recovery per second fixed: recovery = 1 - 0.65 ** (seconds / 10), i.e. 0.19 at 5 s and 0.58
     at 20 s (0.35 at 10 s, so both rules share the reference).
lambda stays 1.0. Greedy-plus-rest is re-tuned for every setting on training people (Stage-1
procedure). The DQN is left out (it was trained with 10-s pauses). The Stage-2 look-ahead
(remaining time / pause seconds, src/learn/al_loop.py) is kept as implemented, so a 20-s pause
gives QMDP a 30-step look-ahead; the pause break-even test documents this effect.

Parts:
  smoke     Tiny version of every part, separate folder. Not a result.
  stage1    Stage 1: validation people, 500 matched sessions; Markov beta 0.5-3 plus FALCON beta 1,
            Lapse beta 1 and no fatigue as controls; all policies except the DQN.
  stage2    Stage 2: development design (validation annotators, seeds 42-51, 50 and 200 starting
            labels), Markov beta 1 and 2 and Lapse beta 1, all 10 policies.
  analyse   Tables, the H2/H3 checks per setting, and the checks of the predictions below.

    python scripts/run_e5_pause_length.py smoke
    python scripts/run_e5_pause_length.py stage1
    python scripts/run_e5_pause_length.py stage2 --workers 13
    python scripts/run_e5_pause_length.py analyse

The predictions are written to outputs/e5_pause_length/predictions.json before any run, and are
never edited afterwards. src/ is not changed. Results go to outputs/e5_pause_length/.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
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
from src.eval.stage1 import benchmark
from src.eval.stats import paired_difference

OUT = project / 'outputs/e5_pause_length'
READ = dict(keep_default_na=False, na_values=[''])
REFERENCE_RECOVERY = .35


def settings():
    """(label, pause seconds, recovery). The 10-s reference is shared by both rules."""
    rows = [('10s_reference', 10., REFERENCE_RECOVERY)]
    for seconds in (5., 20.):
        rows.append((f'{seconds:g}s_A_fixed_per_pause', seconds, REFERENCE_RECOVERY))
        rows.append((f'{seconds:g}s_B_fixed_per_second', seconds, round(1 - (1 - REFERENCE_RECOVERY) ** (seconds / 10), 4)))
    return rows


STAGE1_CASES = {**{f'Markov_beta_{b:g}': Scenario(beta=b) for b in (.5, 1, 2, 3)},
                'FALCON_beta_1': Scenario(model='FALCON', beta=1),
                'Lapse_beta_1': Scenario(model='Lapse', beta=1),
                'no_fatigue': Scenario(model='None', beta=0)}
STAGE1_POLICIES = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload',
                   'blind', 'myopic', 'rest_threshold', 'qmdp', 'qmdp_fixed', 'grid', 'oracle']
REQUIRED_FIXED = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload', 'blind']
STAGE2_POLICIES = ['qmdp', 'qmdp_fixed', 'myopic', 'rest_threshold', 'blind', 'verify', 'two_choice', 'full',
                   'random', 'random_image_full']
MARKOV = ['Markov_beta_0.5', 'Markov_beta_1', 'Markov_beta_2', 'Markov_beta_3']

PREDICTIONS = {
    'basis': 'scripts/pause_break_even.py results (outputs/pause_break_even), written before any E5 run',
    'P1': 'QMDP pauses more per Stage-2 run at 5 s than at 10 s, and less at 20 s than at 10 s (rule A).',
    'P2': 'In Stage 2 Markov, QMDP minus always-verify (average accuracy) is larger at 5 s than at 20 s (rule A).',
    'P3': 'At 20 s (rule A), QMDP minus always-verify in Stage 2 Markov falls to about the E3 format-choice '
          'range (about 0 to +1.1 points).',
    'P4': 'FALCON (Stage 1): QMDP never pauses at any pause length, so its results do not depend on the length.',
    'P5': 'At 5 s, QMDP still roughly ties tuned greedy-plus-rest in Stage 2 Markov (no clear difference).'}


def write_predictions():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'predictions.json'
    if not path.exists():
        path.write_text(json.dumps({**PREDICTIONS, 'written_utc': datetime.now(timezone.utc).isoformat(),
                                    'settings': settings()}, indent=2))


def workers_env():
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'


# ------------------------------------------------------------------ Stage 1
def run_stage1(cases=STAGE1_CASES, episodes=500, seed=42, policies=STAGE1_POLICIES):
    write_predictions()
    contexts = load_contexts()
    tables, paired = [], []
    for label, seconds, recovery in settings():
        for case, base in cases.items():
            scenario = replace(base, pause_seconds=seconds, recovery=recovery)
            table, _ = benchmark(contexts, scenario, policies, episodes=episodes, seed=seed)
            table['setting'], table['case'], table['pause_seconds'], table['recovery_rule'] = label, case, seconds, recovery
            tables.append(table)
            for rival in [p for p in policies if p not in ('qmdp', 'oracle')]:
                paired.append({'setting': label, 'case': case, **paired_difference(table, 'qmdp', rival, seed=seed)})
            print(f'{label} / {case}: done', flush=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(OUT / 'stage1_episodes.csv.gz', index=False)
    pd.DataFrame(paired).round(4).to_csv(OUT / 'stage1_qmdp_vs_rivals.csv', index=False)


# ------------------------------------------------------------------ Stage 2
def stage2_worlds():
    import src.eval.stage2 as stage2
    development = stage2.experiment_cases()   # None, Markov 1, Markov 2, FALCON 1, Lapse 1
    return {'Markov_beta_1': development[1], 'Markov_beta_2': development[2], 'Lapse_beta_1': development[4]}


def run_stage2(workers=6, seeds=range(42, 52), seed_sizes=(50, 200), options=None, worlds=None, chosen=None):
    import src.eval.stage2 as stage2
    write_predictions()
    options = options or stage2.default_options()
    worlds = worlds or stage2_worlds()
    chosen = chosen or settings()
    scenarios = {(label, world): replace(base, pause_seconds=seconds, recovery=recovery)
                 for label, seconds, recovery in chosen for world, base in worlds.items()}
    thresholds = dict(zip(scenarios, stage2.tuned_rest_thresholds(list(scenarios.values()))))
    print('Greedy-plus-rest thresholds:', {f'{k[0]} / {k[1]}': v for k, v in thresholds.items()}, flush=True)
    jobs = [((label, world), (scenario, policy, size, seed,
             {**options, 'rest_threshold': thresholds[(label, world)]} if policy == 'rest_threshold' else options))
            for (label, world), scenario in scenarios.items() for size in seed_sizes for seed in seeds
            for policy in STAGE2_POLICIES]
    workers_env()
    print(f'{len(jobs)} classifier runs on {workers} workers...', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=stage2.initialize_worker) as pool:
        futures = {pool.submit(stage2.learning_job, job): key for key, job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            summary = future.result()
            label, world = futures[future]
            summary.update({'setting': label, 'world': world, 'run_folder': Path(summary['run_folder']).as_posix()})
            rows.append(summary)
            if count % 50 == 0 or count == len(jobs):
                print(f'  {count}/{len(jobs)} done', flush=True)
    table = pd.DataFrame(rows).sort_values(['setting', 'world', 'seed_size', 'seed', 'policy'])
    table.to_csv(OUT / 'stage2_runs.csv', index=False)
    (OUT / 'stage2_settings.json').write_text(json.dumps({'settings': chosen,
        'rest_thresholds': {f'{k[0]} / {k[1]}': v for k, v in thresholds.items()},
        'seeds': list(seeds), 'seed_sizes': list(seed_sizes), 'annotators': 'validation (development)'}, indent=2))


# ------------------------------------------------------------------ analysis
def bootstrap(values, seed=42, resamples=4000):
    values = np.asarray(values, float)
    means = values[np.random.default_rng(seed).integers(len(values), size=(resamples, len(values)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return values.mean(), low, high


def audit_pauses(run_folder):
    audit = pd.read_csv(project / run_folder / 'answer_audit.csv.gz', **READ, usecols=['action', 'accepted', 'fatigue'])
    accepted = audit[audit.accepted.astype(str) == 'True']
    return {'pauses': int((accepted.action == 0).sum()),
            'true_fatigue_when_asking': float(accepted.loc[accepted.action.isin([1, 2, 3]), 'fatigue'].mean())}


def analyse_stage1():
    table = pd.read_csv(OUT / 'stage1_episodes.csv.gz', **READ)
    columns = ['reward', 'information', 'seconds', 'error_rate', 'count_skip', 'count_verify', 'count_two_choice', 'count_full']
    summary = table.groupby(['setting', 'case', 'policy'])[columns].mean()
    summary['bits_per_second'] = summary.information / summary.seconds
    summary.round(3).to_csv(OUT / 'stage1_summary.csv')
    paired = pd.read_csv(OUT / 'stage1_qmdp_vs_rivals.csv', **READ)
    checks = []
    for setting, part in paired.groupby('setting'):
        markov = part[part.case.isin(MARKOV)]
        h2 = markov[markov.second.isin(REQUIRED_FIXED)].groupby('case').ci_low.min() > 0
        gaps = markov[markov.second == 'myopic'].set_index('case').difference.reindex(MARKOV)
        checks.append({'setting': setting, 'markov_h2_passes': int(h2.sum()), 'markov_cases': len(h2),
                       'qmdp_minus_greedy_beta_0.5_1_2_3': gaps.round(2).tolist(),
                       'gap_grows_with_beta': bool(np.all(np.diff(gaps.to_numpy()) > 0)),
                       'falcon_qmdp_pauses_per_session': float(summary.loc[(setting, 'FALCON_beta_1', 'qmdp'), 'count_skip'])
                       if (setting, 'FALCON_beta_1', 'qmdp') in summary.index else np.nan})
    pd.DataFrame(checks).to_csv(OUT / 'stage1_checks.csv', index=False)
    print(pd.DataFrame(checks).to_string(index=False))
    return summary


def analyse_stage2():
    table = pd.read_csv(OUT / 'stage2_runs.csv', **READ)
    focus = table.policy.isin(['qmdp', 'rest_threshold', 'verify'])
    extra = pd.DataFrame([audit_pauses(f) for f in table.loc[focus, 'run_folder']], index=table.index[focus])
    table = table.join(extra)
    # Accuracy at the last budget that exists (the smoke test uses shorter budgets than the study).
    budgets = sorted((c for c in table if c.startswith('accuracy_') and c.endswith('s')), key=lambda c: float(c[9:-1]))
    metrics = [c for c in ['auc_accuracy_percent', budgets[-1], 'acquired_answers', 'acquired_error_rate',
               'answers_after_lapse', 'pauses', 'true_fatigue_when_asking'] if c in table]
    table.groupby(['setting', 'world', 'seed_size', 'policy'])[metrics].mean().round(3).to_csv(OUT / 'stage2_summary.csv')
    paired = []
    for (setting, world, size), part in table.groupby(['setting', 'world', 'seed_size']):
        wide = part.pivot(index='seed', columns='policy', values='auc_accuracy_percent')
        for rival in [p for p in STAGE2_POLICIES if p != 'qmdp']:
            mean, low, high = bootstrap(wide['qmdp'] - wide[rival])
            paired.append({'setting': setting, 'world': world, 'seed_size': size, 'rival': rival,
                           'qmdp_minus_rival': mean, 'ci_low': low, 'ci_high': high})
    paired = pd.DataFrame(paired).round(3)
    paired.to_csv(OUT / 'stage2_qmdp_vs_rivals.csv', index=False)
    change = []
    reference = table[table.setting == '10s_reference']
    for (setting, world, size, policy), part in table[table.setting != '10s_reference'].groupby(['setting', 'world', 'seed_size', 'policy']):
        ref = reference[(reference.world == world) & (reference.seed_size == size) & (reference.policy == policy)]
        difference = part.set_index('seed').auc_accuracy_percent - ref.set_index('seed').auc_accuracy_percent
        mean, low, high = bootstrap(difference.dropna())
        change.append({'setting': setting, 'world': world, 'seed_size': size, 'policy': policy,
                       'change_vs_10s': mean, 'ci_low': low, 'ci_high': high})
    pd.DataFrame(change).round(3).to_csv(OUT / 'stage2_change_vs_10s.csv', index=False)
    return table, paired


def check_predictions(stage2_table, paired, stage1_summary):
    markov = stage2_table[stage2_table.world.str.startswith('Markov')]
    pauses = markov[markov.policy == 'qmdp'].groupby('setting').pauses.mean()
    lead = paired[(paired.rival == 'verify') & paired.world.str.startswith('Markov')].groupby('setting').qmdp_minus_rival.mean()
    rest = paired[(paired.rival == 'rest_threshold') & paired.world.str.startswith('Markov') & (paired.setting == '5s_A_fixed_per_pause')]
    falcon = stage1_summary.xs(('FALCON_beta_1', 'qmdp'), level=('case', 'policy')).count_skip if stage1_summary is not None else None
    five, ten, twenty = '5s_A_fixed_per_pause', '10s_reference', '20s_A_fixed_per_pause'
    def available(series, *keys):
        return all(key in series.index for key in keys)
    # A check reads 'not available' (None) when its settings were not run (for example in the smoke test).
    results = {
        'P1': {'qmdp_pauses_per_run': pauses.round(2).to_dict(),
               'holds': bool(pauses[five] > pauses[ten] > pauses[twenty]) if available(pauses, five, ten, twenty) else None},
        'P2': {'qmdp_minus_verify_mean_over_markov_groups': lead.round(3).to_dict(),
               'holds': bool(lead[five] > lead[twenty]) if available(lead, five, twenty) else None},
        'P3': {'qmdp_minus_verify_20s_A': float(lead[twenty]) if available(lead, twenty) else None,
               'holds': bool(-.1 <= lead[twenty] <= 1.1) if available(lead, twenty) else None},
        'P4': {'falcon_qmdp_pauses_per_session': falcon.round(3).to_dict() if falcon is not None else None,
               'holds': bool((falcon == 0).all()) if falcon is not None else None},
        'P5': {'qmdp_minus_rest_5s_A': rest[['world', 'seed_size', 'qmdp_minus_rival', 'ci_low', 'ci_high']].to_dict('records'),
               'holds': bool(((rest.ci_low <= 0) & (rest.ci_high >= 0)).all()) if len(rest) else None}}
    (OUT / 'prediction_checks.json').write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


def analyse():
    stage1_summary = analyse_stage1() if (OUT / 'stage1_episodes.csv.gz').exists() else None
    if (OUT / 'stage2_runs.csv').exists():
        table, paired = analyse_stage2()
        check_predictions(table, paired, stage1_summary)


# ------------------------------------------------------------------ smoke
def smoke():
    """Tiny version of every part, separate folder. Not a result."""
    global OUT
    OUT = project / 'outputs/e5_pause_length_smoke'
    run_stage1(cases={'Markov_beta_1': Scenario(beta=1), 'FALCON_beta_1': Scenario(model='FALCON', beta=1)},
               episodes=40, policies=['verify', 'myopic', 'rest_threshold', 'qmdp', 'oracle'])
    import src.eval.stage2 as stage2
    options = {**stage2.default_options(), 'budgets_seconds': [150, 300]}
    run_stage2(workers=2, seeds=(42,), seed_sizes=(200,), options=options,
               worlds={'Markov_beta_1': stage2.experiment_cases()[1]}, chosen=settings()[:2])
    analyse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['smoke', 'stage1', 'stage2', 'analyse'])
    parser.add_argument('--workers', type=int, default=6)
    arguments = parser.parse_args()
    {'smoke': smoke, 'stage1': run_stage1, 'stage2': lambda: run_stage2(arguments.workers),
     'analyse': analyse}[arguments.part]()
