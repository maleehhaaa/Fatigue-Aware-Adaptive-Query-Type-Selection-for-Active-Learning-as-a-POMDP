"""Paired development experiments for actual classifier accuracy versus human time."""
from dataclasses import replace
import hashlib
import json
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from src.common import ROOT, save_run, load_calibration, load_contexts
from src.sim.interactive import make_session_factory
from src.sim.fatigue import Scenario
from src.eval.stage1 import population_model, tune_rest_threshold
from src.control.qmdp import PlanningModel
from src.learn.al_loop import run_learning


def default_options():
    return {'budgets_seconds': [300, 600, 1200, 2400], 'session_seconds': 600,
        'retrain_every_answers': 10, 'initial_epochs': 200, 'update_epochs': 10,
        'initial_lr': .05, 'update_lr': .01, 'targets_percent': [75., 80.],
        'seed_label_cost': 'excluded; identical trusted starting labels within each pair',
        'session_rule': 'fixed 600-human-second shifts; crossing answers censored',
        'practice_rule': 'count answered questions, capped at 200 per annotator',
        'evaluation': 'development validation images and validation annotators only',
        'selection': 'QMDP provisional from Notebook 08; full Stage-1 grid pending',
        'rest_threshold': 2.0}  # fallback for single runs; the study tunes it per setting (tuned_rest_thresholds)


def load_learning_data():
    images = pd.read_csv(ROOT / 'data/splits/images.csv').set_index('image_id')
    roles = pd.read_csv(ROOT / 'data/splits/validation_roles.csv')
    with np.load(ROOT / 'data/features/resnet18_development_features.npz') as saved:
        ids = saved['image_ids']
        features = F.normalize(torch.from_numpy(saved['features']), dim=1)
    row_for_id = np.full(10000, -1)
    row_for_id[ids] = np.arange(len(ids))
    calibration_ids = roles.loc[roles.validation_role == 'calibration', 'image_id'].to_numpy()
    evaluation_ids = roles.loc[roles.validation_role == 'evaluation', 'image_id'].to_numpy()
    difficulty = pd.read_csv(ROOT / 'data/processed/predicted_difficulty.csv').set_index('image_id')
    predicted_d = np.full(10000, np.nan)
    predicted_d[difficulty.index] = difficulty.predicted_d
    data = {'features': features, 'row_for_id': row_for_id, 'predicted_d': predicted_d,
        'pool_ids': images.index[images.image_split == 'pool'].to_numpy(),
        'calibration_ids': calibration_ids, 'evaluation_ids': evaluation_ids,
        'calibration_labels': torch.tensor(images.loc[calibration_ids, 'true_label'].to_numpy()),
        'evaluation_labels': torch.tensor(images.loc[evaluation_ids, 'true_label'].to_numpy())}
    assert not np.isin(data['pool_ids'], np.r_[calibration_ids, evaluation_ids]).any()
    assert (row_for_id[images.index[images.image_split == 'final_test']] == -1).all()
    return data


def sample_curve(curve, budgets):
    return np.array([curve.loc[curve.seconds <= budget + 1e-9, 'accuracy_percent'].iloc[-1] for budget in budgets])


def curve_auc(curve, budget):
    points = curve[curve.seconds < budget]
    times = np.r_[points.seconds.to_numpy(), budget]
    return float(np.sum(np.diff(times) * points.accuracy_percent.to_numpy()) / budget)


def run_one(scenario, policy, seed_size=200, seed=42, options=None, data=None, calibration=None):
    torch.set_num_threads(1)
    options = default_options() if options is None else options
    data = load_learning_data() if data is None else data
    calibration = load_calibration() if calibration is None else calibration
    config = {'scenario': scenario.dictionary(), 'policy': policy, 'seed_size': seed_size, 'options': options,
        'source_sha256': hashlib.sha256(b''.join(path.read_bytes() for path in sorted((ROOT / 'src').rglob('*.py')))).hexdigest(),
        'features_metadata_sha256': hashlib.sha256((ROOT / 'data/features/resnet18_development_metadata.json').read_bytes()).hexdigest(),
        'calibration_sha256': hashlib.sha256(b''.join(calibration[key].tobytes() for key in ['theta','mu','sigma','difficulty','image_errors'])).hexdigest(),
        'roles_sha256': hashlib.sha256((ROOT / 'data/splits/validation_roles.csv').read_bytes()).hexdigest()}
    folder = save_run('stage2_development', config, seed)
    if (folder / 'summary.json').exists():
        return json.loads((folder / 'summary.json').read_text()), folder
    seed_ids = np.random.default_rng(seed).permutation(data['pool_ids'])[:seed_size]
    # Only these starting labels leave the simulator; all later training targets come from answers.
    seed_labels = calibration['true_labels'][seed_ids]
    factory, audit = make_session_factory(calibration, scenario, seed)
    start = time.perf_counter()
    result = run_learning(data, seed_ids, seed_labels, scenario, policy, seed, options, population_model(calibration), factory)
    curve, events = result['curve'], result['events']
    audit_table = pd.DataFrame(audit)
    events = events.merge(audit_table, on=['call','session'], validate='one_to_one')
    used = events.accepted & events.action.isin([1,2,3])
    answered = events.accepted & events.action.isin([1,2,3,4])
    summary = {'policy': policy, 'model': scenario.model, 'beta': scenario.beta,
        'seed_size': seed_size, 'seed': seed, 'initial_accuracy': float(curve.accuracy_percent.iloc[0]),
        'auc_accuracy_percent': curve_auc(curve, max(options['budgets_seconds'])),
        'acquired_answers': int(used.sum()), 'acquired_error_rate': float((~events.loc[used,'correct']).mean()) if used.any() else 0.,
        'temperature_boundary': result['learner'].temperature_boundary,
        'pending_answers': result['pending_answers'], 'pool_exhausted': result['pool_exhausted'],
        'spent_seconds': result['stop_seconds'], 'computer_seconds': time.perf_counter() - start,
        'sessions_used': int(events.session.nunique()), 'final_test_evaluated': False,
        # Review additions: how well fatigue was tracked, and whether lapses were actually tested.
        'tracking_mae': float((events.loc[answered, 'estimated_fatigue'] - events.loc[answered, 'fatigue'].clip(0, 4)).abs().mean()) if answered.any() else float('nan'),
        'answers_after_lapse': int((used & events.lapsed).sum())}
    for budget, accuracy in zip(options['budgets_seconds'], sample_curve(curve, options['budgets_seconds'])):
        summary[f'accuracy_{budget}s'] = float(accuracy)
    for target in options['targets_percent']:
        reached = curve[curve.accuracy_percent >= target]
        summary[f'reached_{target:g}'] = not reached.empty
        summary[f'time_{target:g}_capped'] = float(reached.seconds.iloc[0]) if not reached.empty else float(max(options['budgets_seconds']))
    curve.to_csv(folder / 'accuracy_curve.csv', index=False)
    events.to_csv(folder / 'answer_audit.csv.gz', index=False)
    np.savez_compressed(folder / 'training_targets.npz', image_ids=result['training_ids'], allowed=result['targets'], weights=result['weights'])
    torch.save({'state_dict': result['learner'].head.state_dict(), 'temperature': result['learner'].temperature}, folder / 'final_classifier.pt')
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary, folder


def experiment_cases():
    # Review fix: Lapse now forces one lapse per 600-s shift, placed after 100-300 answered
    # questions (mid-shift), so the setting is no longer a copy of Markov beta=1.
    return [Scenario(model='None', beta=0), Scenario(beta=1), Scenario(beta=2),
            Scenario(model='FALCON', beta=1), Scenario(model='Lapse', beta=1,
                lapse_onset_min=100., lapse_onset_max=300.)]


# Added 21 Nov: a second Lapse world whose onset is counted in human seconds, not answers.
# Range: when always-verify reaches its 100th and 300th answer of a 600-s shift in the
# answer-count Lapse world (development study at lambda = 1.0: about 75 s and 310 s), so the
# lambda-free reference meets about the same lapse in both worlds. experiment_cases() is unchanged.
LAPSE_SECONDS_ONSET = (75., 310.)


def lapse_seconds_case():
    return Scenario(model='Lapse', beta=1, lapse_mode='forced_seconds',
                    lapse_onset_min=LAPSE_SECONDS_ONSET[0], lapse_onset_max=LAPSE_SECONDS_ONSET[1])


def final_test_cases():
    """Settings for the final test: the five development settings plus the seconds-onset Lapse world."""
    return experiment_cases() + [lapse_seconds_case()]


def tuned_rest_thresholds(scenarios, seed=42):
    """Greedy-plus-rest pause threshold for each setting, tuned exactly as in Stage 1.

    Review fix (21 Nov): Stage 2 used a fixed threshold of 2.0 in every setting, while Stage 1
    tunes it per setting. Tuning uses training people and the Stage-1 score only (never
    Stage-2 accuracy), so the baseline gets the same treatment in both stages.
    """
    contexts, calibration = load_contexts(), load_calibration()
    population = population_model(calibration)
    return [float(tune_rest_threshold(contexts, scenario, PlanningModel(contexts, scenario, population),
                                      calibration, seed=seed)) for scenario in scenarios]


def paired_accuracy(table, metric, first, second, seed=42):
    paired = table.pivot(index='seed', columns='policy', values=metric)
    difference = (paired[first] - paired[second]).to_numpy()
    random = np.random.default_rng(seed)
    means = difference[random.integers(len(difference), size=(2000, len(difference)))].mean(1)
    low, high = np.quantile(means, [.025,.975])
    return {'first': first, 'second': second, 'metric': metric, 'difference': float(difference.mean()),
        'ci_low': float(low), 'ci_high': float(high), 'pairs': len(difference)}


def initialize_worker():
    global WORKER_DATA, WORKER_CALIBRATION
    torch.set_num_threads(1)
    WORKER_DATA, WORKER_CALIBRATION = load_learning_data(), load_calibration()


def learning_job(job):
    scenario, policy, size, seed, options = job
    summary, folder = run_one(scenario, policy, size, seed, options, WORKER_DATA, WORKER_CALIBRATION)
    summary['run_folder'] = str(folder.relative_to(ROOT))
    return summary


def run_stage2_study(options=None, seed_sizes=(50,200), seeds=range(42,52), workers=6):
    options = default_options() if options is None else options
    policies = ['qmdp','qmdp_fixed','myopic','rest_threshold','blind','verify','two_choice','full','random','random_image_full']
    selected = json.loads((ROOT / 'outputs/stage1_latest.json').read_text())['assessment']['controller']
    assert selected == 'qmdp', 'The Stage-1 choice changed; update the Stage-2 policy adapter before comparing.'
    cases = experiment_cases()
    thresholds = tuned_rest_thresholds(cases)
    config = {'options': options, 'seed_sizes': list(seed_sizes), 'seeds': list(seeds), 'policies': policies,
        'scenarios': [scenario.dictionary() for scenario in cases],
        # One tuned greedy-plus-rest threshold per scenario, in the same order.
        'rest_thresholds': thresholds,
        'source_sha256': hashlib.sha256(b''.join(path.read_bytes() for path in sorted((ROOT / 'src').rglob('*.py')))).hexdigest(),
        'data_sha256': hashlib.sha256((ROOT / 'data/processed/simulator_calibration.npz').read_bytes()
            + (ROOT / 'data/features/resnet18_development_metadata.json').read_bytes()
            + (ROOT / 'data/splits/validation_roles.csv').read_bytes()).hexdigest()}
    folder = save_run('stage2_study', config)
    jobs = [(scenario, policy, size, seed, {**options, 'rest_threshold': threshold} if policy == 'rest_threshold' else options)
            for scenario, threshold in zip(cases, thresholds)
            for size in seed_sizes for seed in seeds for policy in policies]
    print('Greedy-plus-rest thresholds:', {f'{s.model} beta={s.beta:g}': t for s, t in zip(cases, thresholds)}, flush=True)
    # Six independent numerical workers; each uses one CPU thread.
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = '1'
    summaries = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        futures = [pool.submit(learning_job, job) for job in jobs]
        for future in as_completed(futures):
            summaries.append(future.result())
            if len(summaries) % 20 == 0 or len(summaries) == len(jobs):
                print(f'Completed {len(summaries)}/{len(jobs)} classifier runs', flush=True)
    table = pd.DataFrame(summaries).sort_values(['model','beta','seed_size','seed','policy']).reset_index(drop=True)
    table.to_csv(folder / 'all_runs.csv', index=False)
    return table, folder


def comparison_tables(table):
    """Paired accuracy at each budget, and time to 80% only where both runs reached it.

    Review fix: the old time rows averaged time_80_capped, where 2,400 s stands in for
    "never reached", as if it were an observed time.
    """
    accuracy_rows, time_rows = [], []
    for (model, beta, size), group in table.groupby(['model','beta','seed_size']):
        for other in sorted(set(group.policy) - {'qmdp'}):
            for budget in [300,600,1200,2400]:
                result = paired_accuracy(group, f'accuracy_{budget}s', 'qmdp', other)
                accuracy_rows.append({'model':model,'beta':beta,'seed_size':size,'budget':budget,**result})
            reached = group.pivot(index='seed', columns='policy', values='reached_80').astype(bool)
            times = group.pivot(index='seed', columns='policy', values='time_80_capped')
            both = reached['qmdp'] & reached[other]
            saved = (times[other] - times['qmdp'])[both].to_numpy()
            row = {'model':model,'beta':beta,'seed_size':size,'first':other,'second':'qmdp',
                'qmdp_reached':int(reached['qmdp'].sum()),'other_reached':int(reached[other].sum()),
                'pairs_both_reached':int(both.sum()),'seconds_saved_mean':float(saved.mean()) if len(saved) else float('nan'),
                'ci_low':float('nan'),'ci_high':float('nan')}
            if len(saved) >= 2:
                random = np.random.default_rng(42)
                means = saved[random.integers(len(saved), size=(2000, len(saved)))].mean(1)
                row['ci_low'], row['ci_high'] = map(float, np.quantile(means, [.025, .975]))
            time_rows.append(row)
    return pd.DataFrame(accuracy_rows), pd.DataFrame(time_rows)


def run_lambda_check(lams=(.1, .25, .5), scenarios=None, seed_sizes=(200,), seeds=range(42,47),
                     policies=('qmdp', 'two_choice', 'verify'), options=None, workers=6):
    """Review item 13: with no fatigue, does a different lambda make QMDP learn as fast as 2-choice?

    Development data only. Fixed-format baselines ignore lambda, so they run once at the default.
    """
    options = default_options() if options is None else options
    scenarios = scenarios or [Scenario(model='None', beta=0), Scenario(beta=1)]
    jobs = []
    for scenario in scenarios:
        for size in seed_sizes:
            for seed in seeds:
                for policy in policies:
                    for lam in (lams if policy in ['qmdp', 'qmdp_fixed', 'myopic'] else (scenario.lam,)):
                        jobs.append((replace(scenario, lam=lam), policy, size, seed, options))
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = '1'
    summaries = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        futures = {pool.submit(learning_job, job): job for job in jobs}
        for future in as_completed(futures):
            summary = future.result()
            summary['lam'] = futures[future][0].lam
            summaries.append(summary)
    return pd.DataFrame(summaries)
