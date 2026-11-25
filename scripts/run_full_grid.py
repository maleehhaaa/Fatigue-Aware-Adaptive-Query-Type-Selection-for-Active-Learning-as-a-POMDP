"""Full Stage-1 grid (7,812 conditions) and the controller selection rule, in parallel.

Runs exactly what notebook 08 runs with RUN_FULL_GRID = True: the same 7,812 conditions
(src/eval/study.full_cases()), the same 15 policies, 500 matched validation sessions per policy per
condition, seed 42, and the DQN named in outputs/dqn_latest.json. Each condition goes through
src/eval/stage1.benchmark(), so it is saved in the normal Stage-1 run cache (outputs/stage1/...):
a stopped run resumes where it left off, and notebook 08 could reuse the same results.

Differences from the notebook, all for speed and memory only:
  - conditions run in parallel (one CPU thread per worker);
  - the 58-million-row table of all sessions is never built. Each worker returns per-condition means,
    the paired comparisons, and the per-session rewards of the three candidate controllers;
  - the selection rule is applied with the project's own select_controller(full=True) on a table that
    holds only the QMDP, grid and DQN sessions (the only rows the rule reads).

It refuses to run if src/ no longer matches configs/frozen_src.json.

    python scripts/run_full_grid.py smoke                 # 6 conditions, 40 sessions; separate folder
    python scripts/run_full_grid.py run --workers 13      # the full grid (resumable)
    python scripts/run_full_grid.py summarise             # rebuild the summary files from the cache

Results: outputs/full_grid/<config-hash>/seed-42/ (condition_summary.csv, paired_comparisons.csv,
candidate_scores.csv, selection.json, assessment.json, h3_by_beta.csv, run_info.json).
The initial study and outputs/stage1_latest.json are not changed.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
from src.common import load_contexts, save_run
from src.eval.stage1 import benchmark
from src.eval.stats import paired_difference
from src.eval.study import full_cases, select_controller

POLICIES = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload', 'blind',
            'myopic', 'rest_threshold', 'qmdp', 'qmdp_fixed', 'grid', 'dqn', 'oracle']   # as in run_study
REQUIRED = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload', 'blind']  # B1-B8
COMPARISONS = REQUIRED + ['myopic', 'rest_threshold', 'qmdp_fixed']                  # as in comparison_table
CANDIDATES = ['qmdp', 'grid', 'dqn']
MEASURES = ['reward', 'information', 'seconds', 'error_rate', 'questions', 'tracking_error',
            'count_skip', 'count_verify', 'count_two_choice', 'count_full', 'count_gold']
STATE = {}


def src_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted((project / 'src').rglob('*.py')))).hexdigest()


def check_frozen():
    frozen = json.loads((project / 'configs/frozen_src.json').read_text())['src_sha256']
    if src_hash() != frozen:
        raise SystemExit(f'src/ changed since it was frozen ({src_hash()[:12]} vs {frozen[:12]}). Stop and check.')
    return frozen


def load_dqn():
    import torch
    from src.control.dqn import make_network
    info = json.loads((project / 'outputs/dqn_latest.json').read_text())
    folder = project / info['folder'].replace('\\', '/')
    network = make_network()
    network.load_state_dict(torch.load(folder / 'controller.pt', weights_only=True))
    network.eval()
    digest = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in network.state_dict().values())).hexdigest()
    return network, str(folder.relative_to(project)), digest


def initialize_worker(policies):
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    STATE['contexts'] = load_contexts()
    STATE['policies'] = policies
    STATE['dqn'] = None
    if 'dqn' in policies:
        import torch
        torch.set_num_threads(1)
        STATE['dqn'] = load_dqn()[0]


def condition_job(job):
    """One condition: run (or load from cache) and return compact results."""
    name, scenario, purpose, episodes, seed = job
    table, folder = benchmark(STATE['contexts'], scenario, STATE['policies'], episodes=episodes, seed=seed, dqn=STATE['dqn'])
    columns = [c for c in MEASURES if c in table]
    means = table.groupby('policy')[columns].mean().reset_index()
    settings = scenario.dictionary()
    means = means.assign(case=name, purpose=purpose, **settings)
    pairs = []
    for candidate in [c for c in CANDIDATES if c in STATE['policies']]:
        for other in [p for p in COMPARISONS + CANDIDATES if p != candidate and p in STATE['policies']]:
            pairs.append({'case': name, 'model': scenario.model, 'beta': scenario.beta, **paired_difference(table, candidate, other)})
    sessions = {c: table[table.policy == c].sort_values('episode').reward.to_numpy() for c in CANDIDATES if c in STATE['policies']}
    return name, means, pd.DataFrame(pairs), sessions, str(folder.relative_to(project))


def run(workers=6, episodes=500, seed=42, cases=None, policies=POLICIES, out_stage='full_grid'):
    frozen = check_frozen()
    cases = cases if cases is not None else full_cases()
    contexts = load_contexts()
    dqn_folder, dqn_digest = (load_dqn()[1:] if 'dqn' in policies else (None, None))
    config = {'cases': len(cases), 'episodes': episodes, 'seed': seed, 'policies': policies, 'src_sha256': frozen,
              'dqn_folder': dqn_folder, 'dqn_sha256': dqn_digest,
              'contexts_sha256': hashlib.sha256(contexts['probabilities'].tobytes() + contexts['difficulty'].tobytes()).hexdigest(),
              'rule': 'ProjectBrief 8.4 via src.eval.study.select_controller(full=True)'}
    folder = save_run(out_stage, config, seed)
    jobs = [(name, scenario, purpose, episodes, seed) for name, scenario, purpose in cases]
    print(f'{len(jobs)} conditions x {len(policies)} policies x {episodes} sessions on {workers} workers. Output: {folder}', flush=True)
    means, pairs, sessions, caches = [], [], {}, {}
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(policies,)) as pool:
        futures = [pool.submit(condition_job, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            name, mean, pair, session, cache = future.result()
            means.append(mean), pairs.append(pair)
            sessions[name], caches[name] = session, cache
            if count % 25 == 0 or count == len(jobs):
                elapsed = time.time() - start
                eta = elapsed / count * (len(jobs) - count)
                print(f'  {count}/{len(jobs)} conditions | {elapsed / 60:.1f} min elapsed | about {eta / 60:.0f} min left', flush=True)
    condition_summary = pd.concat(means, ignore_index=True)
    condition_summary.to_csv(folder / 'condition_summary.csv', index=False)
    pd.concat(pairs, ignore_index=True).to_csv(folder / 'paired_comparisons.csv', index=False)
    np.savez_compressed(folder / 'candidate_sessions.npz', **{f'{name}|{c}': v for name, s in sessions.items() for c, v in s.items()})
    (folder / 'cache_folders.json').write_text(json.dumps(caches, indent=2))
    (folder / 'run_info.json').write_text(json.dumps({**config, 'finished_utc': datetime.now(timezone.utc).isoformat(),
                                                     'minutes': round((time.time() - start) / 60, 1), 'workers': workers}, indent=2))
    summarise(folder, cases)
    return folder


def summarise(folder=None, cases=None):
    """Selection rule and assessment from the saved per-condition files (no reruns)."""
    if folder is None:
        candidates = sorted((project / 'outputs/full_grid').glob('*/seed-42/condition_summary.csv'), key=lambda p: p.stat().st_mtime)
        folder = candidates[-1].parent
    folder = Path(folder)
    read = dict(keep_default_na=False, na_values=[''])
    summary = pd.read_csv(folder / 'condition_summary.csv', **read)
    pairs = pd.read_csv(folder / 'paired_comparisons.csv', **read)
    sessions = np.load(folder / 'candidate_sessions.npz')
    model_of = summary.drop_duplicates('case').set_index('case')[['model', 'purpose']]
    # The rule reads only the candidate controllers' sessions: build just that table.
    rows = []
    for key in sessions.files:
        case, policy = key.split('|')
        values = sessions[key]
        rows.append(pd.DataFrame({'policy': policy, 'episode': np.arange(len(values)), 'reward': values,
                                  'model': model_of.loc[case, 'model'], 'purpose': model_of.loc[case, 'purpose']}))
    candidate_table = pd.concat(rows, ignore_index=True)
    # model and purpose are only filtered on, so they can be compact; policy stays plain text, so the
    # rule's groupby behaves exactly as in notebook 08.
    for column in ['model', 'purpose']:
        candidate_table[column] = candidate_table[column].astype('category')
    winner, scores = select_controller(candidate_table, full=True)
    scores.to_csv(folder / 'candidate_scores.csv')
    selection = {'controller': winner, 'status': 'full-grid validation selection (ProjectBrief 8.4)',
                 'completed_conditions': int(summary.case.nunique()), 'full_grid_conditions': len(full_cases()),
                 'scores': scores.round(4).to_dict('index'),
                 'test_annotators_evaluated': False, 'final_test_images_evaluated': False}
    (folder / 'selection.json').write_text(json.dumps(selection, indent=2))
    # Assessment for the selected controller (same definitions as notebook 08).
    mine = pairs[pairs['first'] == winner]
    required = mine[mine.second.isin(REQUIRED)]
    beats = required.groupby(['case', 'model', 'beta']).ci_low.min().gt(0).rename('beats_all_required').reset_index()
    beats.to_csv(folder / 'h2_by_condition.csv', index=False)
    h2 = beats[beats.beta >= 1].groupby('model').beats_all_required.agg(['sum', 'count'])
    greedy = mine[mine.second == 'myopic'].merge(summary[summary.policy == winner][['case', 'distribution', 'recovery', 'signal',
        'verify_time', 'choice_time', 'steps']], on='case')
    by_beta = greedy[greedy.model != 'None'].groupby(['model', 'beta']).difference.agg(['mean', 'count']).reset_index()
    by_beta.to_csv(folder / 'h3_by_beta.csv', index=False)
    keys = ['model', 'distribution', 'recovery', 'signal', 'verify_time', 'choice_time', 'steps']
    grows = (greedy[greedy.model.isin(['Markov', 'FALCON'])].sort_values('beta').groupby(keys).difference
             .apply(lambda d: bool(np.all(np.diff(d.to_numpy()) > 0)) if len(d) == 4 else np.nan)   # all four beta values
             .dropna().groupby('model').mean())
    none = summary[summary.model == 'None'].pivot_table(index='case', columns='policy', values='reward')
    none_gap = (none[winner] - none[REQUIRED].max(axis=1))
    assessment = {'controller': winner,
                  'h2_beats_all_B1_B8_beta_ge_1': {m: f"{int(r['sum'])}/{int(r['count'])}" for m, r in h2.iterrows()},
                  'h3_mean_gap_over_greedy_by_beta': {f"{r['model']} beta={r['beta']:g}": round(float(r['mean']), 3)
                                                      for _, r in by_beta.iterrows()},
                  'h3_share_of_settings_where_gap_grows_with_beta': grows.round(3).to_dict(),
                  'h1_no_fatigue_minus_best_required_baseline': {'mean': round(float(none_gap.mean()), 4),
                                                                 'min': round(float(none_gap.min()), 4),
                                                                 'max': round(float(none_gap.max()), 4)},
                  'note': 'Validation people only; pointwise unadjusted intervals; no test annotators or final-test images.'}
    (folder / 'assessment.json').write_text(json.dumps(assessment, indent=2))
    print(json.dumps(selection, indent=2))
    print(json.dumps(assessment, indent=2))
    if winner != 'qmdp':
        print('\nNOTE: the selected controller is not QMDP. The Stage-2 code asserts QMDP; decide before continuing.')


def smoke(workers=2, policies=POLICIES):
    """Six conditions spread over the grid, 40 sessions, separate output folder. Not a result."""
    cases = full_cases()
    picks = [cases[i] for i in np.linspace(0, len(cases) - 1, 6).astype(int)]
    run(workers=workers, episodes=40, cases=picks, policies=policies, out_stage='full_grid_smoke')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['run', 'summarise', 'smoke'])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--folder', help='for summarise: a specific outputs/full_grid/<id>/seed-42 folder')
    arguments = parser.parse_args()
    if arguments.part == 'run':
        run(arguments.workers)
    elif arguments.part == 'smoke':
        smoke(arguments.workers)
    else:
        summarise(project / arguments.folder if arguments.folder else None)
