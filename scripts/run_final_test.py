"""Final test: run ONCE, with settings fixed in outputs/final_test/settings.json before any run.

What is tested:
  Stage 1  Test annotators, seed 2026, 500 matched sessions, all 15 policies (incl. grid and DQN):
           the 33 cases of src/eval/study.final_test_cases() (28 initial + 5 seconds-onset Lapse)
           and the 6 robustness cases of notebook 08 (QMDP planned for Markov beta = 2).
  Stage 2  Test annotators, all 2,000 final-test images for scoring (the 1,000 validation
           calibration images still set the confidence temperature; same 4,000-image pool and
           starting-label rule), seeds 1000-1019, 50 and 200 starting labels, the 6 settings of
           src/eval/stage2.final_test_cases(), all 10 policies, budgets 300/600/1200/2400 s.
  Grid     (additional, descriptive) the full 7,812-condition grid on test annotators, seed 2026.
Everything else is frozen at the development settings: lambda = 1.0, 10-s pause, greedy-plus-rest
re-tuned on training people, frozen src/ (configs/frozen_src.json), controller from the full-grid
selection (QMDP).

Statistics: paired 95% bootstrap intervals (4,000 resamples, seed 42) and two-sided sign-flip
permutation p-values (exact for up to 20 pairs; 20,000 random flips for 500 sessions). Holm
correction, one family per comparison type: Stage 2, for each rival across the 12 groups; Stage 1,
for each rival across the cases of each family. The primary Stage-2 test in each group is QMDP
against a rival named in advance from the development results (PRIMARY_RIVALS); the 12 primary
tests are also corrected together as one family. Unadjusted intervals are reported next to
Holm-adjusted p-values.

Parts:
  features   Extract ResNet-18 features of the 2,000 final-test images (notebook 02 preprocessing),
             after checking that the same code reproduces saved development features.
  settings   Write outputs/final_test/settings.json (refuses to change it once written).
  smoke      Exercise every code path with VALIDATION annotators, VALIDATION evaluation images and
             smoke seeds (9000+), into outputs/final_test_smoke. Never touches test annotators, the
             final-test labels or the real seeds.
  stage1                 Final Stage 1.
  stage2 --workers 13    Final Stage 2.
  grid --workers 13      Full grid on test annotators (additional).
  analyse                Tables, tests and Holm corrections; outputs/final_test/final_report.json.

Each of stage1 / stage2 / grid runs once: it refuses to start if it has already completed. If it
was interrupted, rerunning resumes from the run cache with the same settings (results are
deterministic, so resuming changes nothing). src/ is not changed.
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
import time

import numpy as np
import pandas as pd

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
sys.path.insert(0, str(project / 'scripts'))
from src.common import load_contexts
from src.sim.fatigue import Scenario
from src.eval.stage1 import benchmark
from src.eval.stats import paired_difference
import src.eval.study as study
from run_full_grid import POLICIES, REQUIRED, check_frozen, load_dqn

OUT = project / 'outputs/final_test'
SMOKE_OUT = project / 'outputs/final_test_smoke'
READ = dict(keep_default_na=False, na_values=[''])
FEATURES = project / 'data/features/resnet18_final_test_features.npz'
FEATURES_META = project / 'data/features/resnet18_final_test_metadata.json'
FULL_GRID = project / 'outputs/full_grid/2954f9bca0/seed-42'
THREADS = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']

# ---------------------------------------------------------------- fixed choices (agreed 22 Nov)
STAGE1_SPLIT, STAGE1_SEED, STAGE1_SESSIONS = 'test', 2026, 500
STAGE2_SEEDS, STAGE2_SIZES = tuple(range(1000, 1020)), (50, 200)
STAGE2_POLICIES = ['qmdp', 'qmdp_fixed', 'myopic', 'rest_threshold', 'blind', 'verify', 'two_choice', 'full',
                   'random', 'random_image_full']
ROBUSTNESS_POLICIES = ['qmdp', 'myopic', 'blind', 'verify']
FINAL_LABEL = 'FINAL TEST: test annotators; 2,000 final-test images for scoring'
# Named in advance from the development study (strongest non-QMDP rival per setting, both sizes).
PRIMARY_RIVALS = {'None0': 'blind', 'Markov1': 'rest_threshold', 'Markov2': 'rest_threshold',
                  'FALCON1': 'myopic', 'Lapse1:forced': 'blind', 'Lapse1:forced_seconds': 'two_choice'}
STAGE1_RIVALS = REQUIRED + ['myopic', 'rest_threshold', 'qmdp_fixed', 'grid', 'dqn']


def robustness_cases():
    """The six robustness conditions of notebook 08: QMDP plans with Markov beta = 2."""
    assumed = Scenario(beta=2)
    return assumed, {'weaker_fatigue': replace(assumed, beta=1), 'stronger_fatigue': replace(assumed, beta=3),
                     'twice_as_fast': replace(assumed, speed=2), 'half_as_fast': replace(assumed, speed=.5),
                     'FALCON_world': replace(assumed, model='FALCON'), 'Lapse_world': replace(assumed, model='Lapse')}


def setting_key(scenario):
    key = f'{scenario.model}{scenario.beta:g}'
    return f'{key}:{scenario.lapse_mode}' if scenario.model == 'Lapse' else key


def stage1_family(purpose, model):
    if purpose == 'core':
        return 'no_fatigue' if model == 'None' else f'core_{model}'
    return purpose      # sensitivity, control, stress, lapse_seconds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- features
def picture_reader():
    import torch
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.hub.set_dir(str(project / 'data/pretrained'))
    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    reader = resnet18(weights=weights)
    reader.fc = nn.Identity()
    reader.requires_grad_(False)
    reader.eval().to(device)
    return reader, preprocess, device


def extract(reader, preprocess, device, pictures, ids, batch_size=64):
    """Same steps as notebook 02, cell 14."""
    import torch
    batches = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            images = torch.from_numpy(pictures.data[batch]).permute(0, 3, 1, 2).float() / 255
            batches.append(reader(preprocess(images).to(device)).cpu().numpy())
    return np.concatenate(batches)


def run_features():
    from torchvision.datasets import CIFAR10
    if FEATURES.exists():
        raise SystemExit(f'{FEATURES.name} already exists; not overwriting it.')
    images = pd.read_csv(project / 'data/splits/images.csv').sort_values('image_id')
    pictures = CIFAR10(root=str(project / 'data/cifar10'), train=False, download=False)
    assert np.array_equal(images.image_id, np.arange(len(pictures)))
    assert np.array_equal(images.true_label, np.array(pictures.targets))
    reader, preprocess, device = picture_reader()
    saved_meta = json.loads((project / 'data/features/resnet18_development_metadata.json').read_text())
    meta = {'weights': 'ResNet18_Weights.IMAGENET1K_V1', 'preprocessing': str(preprocess),
            'image_split_sha256': sha(project / 'data/splits/images.csv')}
    assert meta == saved_meta, 'Preprocessing or splits differ from the development features.'
    # Reproduction check: the same code must give the saved development features.
    with np.load(project / 'data/features/resnet18_development_features.npz') as saved:
        dev_ids, dev_features = saved['image_ids'], saved['features']
    check = np.linspace(0, len(dev_ids) - 1, 256).astype(int)
    again = extract(reader, preprocess, device, pictures, dev_ids[check])
    relative = float(np.abs(again - dev_features[check]).max() / np.abs(dev_features[check]).max())
    print(f'Reproduction check on 256 development images ({device}): max relative difference {relative:.2e}')
    assert relative < 1e-3, 'Cannot reproduce the saved development features; stop and check.'
    final_ids = images.loc[images.image_split == 'final_test', 'image_id'].to_numpy()
    assert len(final_ids) == 2000
    features = extract(reader, preprocess, device, pictures, final_ids)
    assert features.shape == (2000, 512) and np.isfinite(features).all()
    np.savez_compressed(FEATURES, image_ids=final_ids, features=features)
    FEATURES_META.write_text(json.dumps({**meta, 'device': device, 'reproduction_check_max_relative_difference': relative,
                                         'written_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f'Saved {FEATURES.relative_to(project)}: {features.shape}')


def build_learning_data(evaluation):
    """Stage-2 data with the final-test features added. evaluation = 'final_test' (the real test) or
    'validation' (smoke: the usual 1,000 validation evaluation images). Only the scoring set changes."""
    import torch
    from torch.nn import functional as F
    images = pd.read_csv(project / 'data/splits/images.csv').set_index('image_id')
    roles = pd.read_csv(project / 'data/splits/validation_roles.csv')
    with np.load(project / 'data/features/resnet18_development_features.npz') as dev, np.load(FEATURES) as final:
        ids = np.r_[dev['image_ids'], final['image_ids']]
        features = F.normalize(torch.from_numpy(np.r_[dev['features'], final['features']]), dim=1)
    row_for_id = np.full(10000, -1)
    row_for_id[ids] = np.arange(len(ids))
    calibration_ids = roles.loc[roles.validation_role == 'calibration', 'image_id'].to_numpy()
    if evaluation == 'final_test':
        evaluation_ids = images.index[images.image_split == 'final_test'].to_numpy()
    else:
        evaluation_ids = roles.loc[roles.validation_role == 'evaluation', 'image_id'].to_numpy()
    difficulty = pd.read_csv(project / 'data/processed/predicted_difficulty.csv').set_index('image_id')
    predicted_d = np.full(10000, np.nan)
    predicted_d[difficulty.index] = difficulty.predicted_d
    pool_ids = images.index[images.image_split == 'pool'].to_numpy()
    data = {'features': features, 'row_for_id': row_for_id, 'predicted_d': predicted_d, 'pool_ids': pool_ids,
            'calibration_ids': calibration_ids, 'evaluation_ids': evaluation_ids,
            'calibration_labels': torch.tensor(images.loc[calibration_ids, 'true_label'].to_numpy()),
            'evaluation_labels': torch.tensor(images.loc[evaluation_ids, 'true_label'].to_numpy())}
    assert not np.isin(pool_ids, np.r_[calibration_ids, evaluation_ids]).any()
    assert not np.isin(calibration_ids, evaluation_ids).any()
    assert (row_for_id[np.r_[pool_ids, calibration_ids, evaluation_ids]] >= 0).all()
    if evaluation == 'final_test':
        assert len(evaluation_ids) == 2000 and (images.loc[evaluation_ids, 'image_split'] == 'final_test').all()
    return data


# ---------------------------------------------------------------- settings
def frozen_settings():
    import src.eval.stage2 as stage2
    selection = json.loads((FULL_GRID / 'selection.json').read_text())
    assumed, robust = robustness_cases()
    options = {**stage2.default_options(), 'evaluation': FINAL_LABEL, 'annotators': 'test', 'final_test': True}
    return {
        'purpose': 'Final test, run once. Fixed before any final-test run; never edited afterwards.',
        'src_sha256': json.loads((project / 'configs/frozen_src.json').read_text())['src_sha256'],
        'controller': selection['controller'], 'controller_selection': str(FULL_GRID.relative_to(project)),
        'dqn': dict(zip(['folder', 'sha256'], load_dqn()[1:])),
        'final_test_features_sha256': sha(FEATURES) if FEATURES.exists() else None,
        'stage1': {'annotators': STAGE1_SPLIT, 'seed': STAGE1_SEED, 'sessions': STAGE1_SESSIONS, 'policies': POLICIES,
                   'cases': [name for name, _, _ in study.final_test_cases()],
                   'robustness': {'assumed': assumed.dictionary(), 'cases': list(robust), 'policies': ROBUSTNESS_POLICIES}},
        'stage2': {'annotators': 'test', 'seeds': list(STAGE2_SEEDS), 'starting_labels': list(STAGE2_SIZES),
                   'settings': [s.dictionary() for s in stage2.final_test_cases()], 'policies': STAGE2_POLICIES,
                   'scoring_images': 'all 2,000 final-test images', 'calibration_images': '1,000 validation calibration images',
                   'pool': '4,000 pool images', 'options': options,
                   'rest_thresholds': 'tuned per setting on training people with the Stage-1 procedure'},
        'grid': {'annotators': STAGE1_SPLIT, 'seed': STAGE1_SEED, 'sessions': STAGE1_SESSIONS, 'cases': len(study.full_cases()),
                 'use': 'additional; reported as rates, not as individual tests'},
        'statistics': {'intervals': 'paired bootstrap 95%, 4,000 resamples, seed 42',
                       'p_values': 'two-sided sign-flip permutation (exact for <= 20 pairs, else 20,000 flips, seed 42)',
                       'correction': 'Holm; Stage 2: per rival across the 12 groups (and the 12 primary tests as one family); '
                                     'Stage 1: per rival across the cases of each family',
                       'stage2_primary_metric': 'auc_accuracy_percent (average accuracy over 0-2,400 s)',
                       'stage2_primary_rivals': PRIMARY_RIVALS,
                       'h2_rule': 'QMDP beats all 8 B1-B8 baselines in a case, each Holm-adjusted p < 0.05 with a positive difference'}}


def write_settings():
    if not FEATURES.exists():
        raise SystemExit('Extract the final-test features first (part: features).')
    check_frozen()
    settings = frozen_settings()
    if settings['controller'] != 'qmdp':
        raise SystemExit('The full-grid selection is not QMDP; the Stage-2 code supports QMDP only. Stop and decide.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'settings.json'
    if path.exists():
        saved = json.loads(path.read_text())
        same = {k: v for k, v in saved.items() if k != 'written_utc'} == settings
        raise SystemExit('settings.json already written; ' + ('it matches the script.' if same else 'it DIFFERS from the script. Do not run.'))
    path.write_text(json.dumps({**settings, 'written_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f'Wrote {path.relative_to(project)}')


def require_settings():
    check_frozen()
    path = OUT / 'settings.json'
    if not path.exists():
        raise SystemExit('Write the settings first (part: settings).')
    saved = json.loads(path.read_text())
    if {k: v for k, v in saved.items() if k != 'written_utc'} != json.loads(json.dumps(frozen_settings())):
        raise SystemExit('settings.json does not match the script or the frozen inputs. Do not run.')
    return saved


def run_once(part):
    """Use-once guard. Returns a function that marks the part complete."""
    done, started = OUT / f'{part}_complete.json', OUT / f'{part}_started.json'
    if done.exists():
        raise SystemExit(f'The final-test {part} has already been run (see {done.name}). It is run once only.')
    if started.exists():
        print(f'Resuming an interrupted final-test {part} with the same settings (finished runs come from the run cache).')
    else:
        started.write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat()}))

    def finish(info):
        done.write_text(json.dumps({**info, 'finished_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    return finish


# ---------------------------------------------------------------- Stage 1
def run_stage1(out=None, split=STAGE1_SPLIT, seed=STAGE1_SEED, sessions=STAGE1_SESSIONS, cases=None, robust=True,
               policies=POLICIES):
    out = out or OUT
    finish = None
    if out == OUT:
        require_settings()
        finish = run_once('stage1')
    out.mkdir(parents=True, exist_ok=True)
    contexts = load_contexts()
    dqn = load_dqn()[0] if 'dqn' in policies else None
    cases = cases if cases is not None else study.final_test_cases()
    tables = []
    for number, (name, scenario, purpose) in enumerate(cases, 1):
        table, _ = benchmark(contexts, scenario, policies, split=split, episodes=sessions, seed=seed, dqn=dqn)
        tables.append(table.assign(case=name, purpose=purpose, family=stage1_family(purpose, scenario.model),
                                   model=scenario.model, beta=scenario.beta, lapse_mode=scenario.lapse_mode))
        print(f'Stage 1 case {number}/{len(cases)}: {name}', flush=True)
    pd.concat(tables, ignore_index=True).to_csv(out / 'stage1_sessions.csv.gz', index=False)
    if robust:
        assumed, robust_cases = robustness_cases()
        rows = []
        for name, actual in robust_cases.items():
            table, _ = benchmark(contexts, actual, [p for p in ROBUSTNESS_POLICIES if p in policies or p == 'qmdp'],
                                 split=split, episodes=sessions, seed=seed, dqn=dqn, assumed=assumed, stage='robustness')
            rows.append(table.assign(condition=name))
            print(f'Robustness: {name}', flush=True)
        pd.concat(rows, ignore_index=True).to_csv(out / 'robustness_sessions.csv.gz', index=False)
    if finish:
        finish({'cases': len(cases), 'split': split, 'seed': seed, 'sessions': sessions})


# ---------------------------------------------------------------- Stage 2
def initialize_final_worker(evaluation, annotators):
    import src.eval.stage2 as stage2
    stage2.initialize_worker()
    stage2.WORKER_DATA = build_learning_data(evaluation)
    stage2.make_session_factory = partial(stage2.make_session_factory, split=annotators)


def final_job(job):
    import src.eval.stage2 as stage2
    return stage2.learning_job(job)


def run_stage2(workers=6, out=None, evaluation='final_test', annotators='test', seeds=STAGE2_SEEDS, sizes=STAGE2_SIZES,
               settings=None, policies=STAGE2_POLICIES, options=None):
    import src.eval.stage2 as stage2
    out = out or OUT
    finish = None
    if out == OUT:
        saved = require_settings()
        finish = run_once('stage2')
        options = saved['stage2']['options']
    out.mkdir(parents=True, exist_ok=True)
    settings = settings or stage2.final_test_cases()
    thresholds = dict(zip([setting_key(s) for s in settings], stage2.tuned_rest_thresholds(settings)))
    print('Greedy-plus-rest thresholds (training people):', thresholds, flush=True)
    jobs = [(setting_key(s), (s, policy, size, seed,
             {**options, 'rest_threshold': thresholds[setting_key(s)]} if policy == 'rest_threshold' else options))
            for s in settings for size in sizes for seed in seeds for policy in policies]
    for name in THREADS:
        os.environ[name] = '1'
    print(f'{len(jobs)} classifier runs on {workers} workers (scoring: {evaluation}; annotators: {annotators})...', flush=True)
    rows, start = [], time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_final_worker, initargs=(evaluation, annotators)) as pool:
        futures = {pool.submit(final_job, job): key for key, job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            summary = future.result()
            summary.update({'setting': futures[future], 'run_folder': Path(summary['run_folder']).as_posix(),
                            'evaluation_images': evaluation, 'annotators': annotators})
            summary['final_test_evaluated'] = evaluation == 'final_test'
            rows.append(summary)
            if count % 50 == 0 or count == len(jobs):
                print(f'  {count}/{len(jobs)} | {(time.time() - start) / 60:.1f} min', flush=True)
    table = pd.DataFrame(rows).sort_values(['setting', 'seed_size', 'seed', 'policy'])
    table.to_csv(out / 'stage2_runs.csv', index=False)
    (out / 'stage2_rest_thresholds.json').write_text(json.dumps(thresholds, indent=2))
    if finish:
        finish({'runs': len(table), 'rest_thresholds': thresholds})


# ---------------------------------------------------------------- full grid on test annotators
WORKER = {}


def grid_worker(policies):
    for name in THREADS:
        os.environ[name] = '1'
    WORKER['contexts'] = load_contexts()
    WORKER['policies'] = policies
    WORKER['dqn'] = None
    if 'dqn' in policies:
        import torch
        torch.set_num_threads(1)
        WORKER['dqn'] = load_dqn()[0]


def grid_job(job):
    name, scenario, split, sessions, seed = job
    table, _ = benchmark(WORKER['contexts'], scenario, WORKER['policies'], split=split, episodes=sessions, seed=seed,
                         dqn=WORKER['dqn'])
    means = table.groupby('policy').reward.mean().rename(name)
    pairs = [{'case': name, 'model': scenario.model, 'beta': scenario.beta, **paired_difference(table, 'qmdp', other)}
             for other in STAGE1_RIVALS if other in WORKER['policies']]
    return name, means, pairs


def run_grid(workers=6, out=None, split=STAGE1_SPLIT, seed=STAGE1_SEED, sessions=STAGE1_SESSIONS, cases=None,
             policies=POLICIES):
    out = out or OUT
    finish = None
    if out == OUT:
        require_settings()
        finish = run_once('grid')
    out.mkdir(parents=True, exist_ok=True)
    cases = cases if cases is not None else study.full_cases()
    jobs = [(name, scenario, split, sessions, seed) for name, scenario, _ in cases]
    means, pairs, start = [], [], time.time()
    print(f'{len(jobs)} grid conditions on {workers} workers ({split} annotators)...', flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=grid_worker, initargs=(policies,)) as pool:
        futures = [pool.submit(grid_job, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            _, mean, pair = future.result()
            means.append(mean)
            pairs.extend(pair)
            if count % 100 == 0 or count == len(jobs):
                elapsed = time.time() - start
                print(f'  grid {count}/{len(jobs)} | {elapsed / 60:.1f} min | about '
                      f'{elapsed / count * (len(jobs) - count) / 60:.0f} min left', flush=True)
    pd.concat(means, axis=1).T.to_csv(out / 'grid_policy_means.csv')
    pd.DataFrame(pairs).to_csv(out / 'grid_qmdp_vs_rivals.csv', index=False)
    if finish:
        finish({'cases': len(cases)})


# ---------------------------------------------------------------- statistics
def sign_flip_p(differences, flips=20000, seed=42):
    """Two-sided sign-flip permutation p-value for a mean paired difference of zero."""
    d = np.asarray(differences, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0 or np.allclose(d, 0):
        return 1.0
    observed = abs(d.mean())
    if n <= 20:   # exact: all 2^n sign patterns
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        return float((np.abs(signs @ d) / n >= observed - 1e-12).mean())
    signs = np.random.default_rng(seed).choice([-1., 1.], size=(flips, n))
    return float(((np.abs(signs @ d) / n >= observed - 1e-12).sum() + 1) / (flips + 1))


def bootstrap(differences, seed=42, resamples=4000):
    d = np.asarray(differences, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    means = d[np.random.default_rng(seed).integers(len(d), size=(resamples, len(d)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return d.mean(), low, high


def holm(p_values):
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1., running)
    return adjusted


def test_row(differences, **keys):
    d = np.asarray(differences, float)
    mean, low, high = bootstrap(d)
    return {**keys, 'difference': mean, 'ci_low': low, 'ci_high': high, 'p': sign_flip_p(d), 'pairs': int(np.isfinite(d).sum())}


# ---------------------------------------------------------------- analysis
def analyse_stage1(out):
    table = pd.read_csv(out / 'stage1_sessions.csv.gz', **READ)
    columns = ['reward', 'information', 'seconds', 'error_rate', 'count_skip', 'count_verify', 'count_two_choice', 'count_full']
    summary = table.groupby(['family', 'case', 'policy'])[columns].mean()
    summary['bits_per_second'] = summary.information / summary.seconds
    summary.round(4).to_csv(out / 'stage1_summary.csv')
    rows = []
    for (family, case), part in table.groupby(['family', 'case']):
        wide = part.pivot(index='episode', columns='policy', values='reward')
        for rival in [r for r in STAGE1_RIVALS if r in wide]:
            rows.append(test_row((wide['qmdp'] - wide[rival]).to_numpy(), family=family, case=case, rival=rival))
    tests = pd.DataFrame(rows)
    tests['p_holm'] = tests.groupby(['family', 'rival']).p.transform(lambda p: holm(p.to_numpy()))
    tests.round(6).to_csv(out / 'stage1_tests.csv', index=False)
    required = tests[tests.rival.isin(REQUIRED)]
    h2 = (required.groupby(['family', 'case'])
          .apply(lambda g: bool(((g.p_holm < .05) & (g.difference > 0)).all()), include_groups=False)
          .rename('beats_all_B1_B8_holm').reset_index())
    h2 = h2.merge(table.drop_duplicates('case')[['case', 'model', 'beta']], on='case')
    h2.to_csv(out / 'stage1_h2.csv', index=False)
    # H2 as defined in the brief: Markov and FALCON, beta >= 1. Other families are listed for information.
    defined = h2[h2.model.isin(['Markov', 'FALCON']) & (h2.beta >= 1)]
    result = {'h2_markov_falcon_beta_ge_1_holm': {f'{m}': f'{int(g.sum())}/{len(g)}' for m, g in defined.groupby('model').beats_all_B1_B8_holm},
              'all_cases_beating_all_B1_B8_holm_by_family': {f: f'{int(g.sum())}/{len(g)}' for f, g in h2.groupby('family').beats_all_B1_B8_holm}}
    if 'no_fatigue' in set(table.case):
        none = table[table.case == 'no_fatigue'].pivot(index='episode', columns='policy', values='reward')
        best = none[[p for p in REQUIRED if p in none]].mean().idxmax()
        gap = none['qmdp'] - none[best]
        se = float(gap.std(ddof=1) / np.sqrt(len(gap)))
        result['h1'] = {'best_required_baseline': best, 'qmdp_minus_best': float(gap.mean()), 'se': se,
                        'within_1_se_or_better': bool(gap.mean() >= -se)}
    markov = [f'Markov_beta_{b:g}' for b in (.5, 1, 2, 3)]
    if all(c in set(table.case) for c in markov):
        betas = np.array([.5, 1., 2., 3.])
        gaps = np.stack([table[table.case == c].pivot(index='episode', columns='policy', values='reward')
                         .pipe(lambda w: (w['qmdp'] - w['myopic']).to_numpy()) for c in markov])
        draws = np.random.default_rng(42).integers(gaps.shape[1], size=(4000, gaps.shape[1]))
        slopes = np.array([np.polyfit(betas, gaps[:, d].mean(1), 1)[0] for d in draws])
        result['h3_markov'] = {'qmdp_minus_greedy_beta_0.5_1_2_3': gaps.mean(1).round(3).tolist(),
                               'slope_per_beta': float(np.polyfit(betas, gaps.mean(1), 1)[0]),
                               'slope_ci': np.quantile(slopes, [.025, .975]).round(3).tolist()}
    if (out / 'robustness_sessions.csv.gz').exists():
        robust = pd.read_csv(out / 'robustness_sessions.csv.gz', **READ)
        rows = [test_row(part.pivot(index='episode', columns='policy', values='reward')
                         .pipe(lambda w: (w['qmdp'] - w['myopic']).to_numpy()), condition=condition, rival='myopic')
                for condition, part in robust.groupby('condition')]
        robust_tests = pd.DataFrame(rows)
        robust_tests['p_holm'] = holm(robust_tests.p.to_numpy())
        robust_tests.round(6).to_csv(out / 'robustness_tests.csv', index=False)
        result['robustness_beats_greedy_holm'] = \
            f"{int(((robust_tests.p_holm < .05) & (robust_tests.difference > 0)).sum())}/{len(robust_tests)}"
    return result


def analyse_stage2(out):
    table = pd.read_csv(out / 'stage2_runs.csv', **READ)
    for target in ['75', '80']:
        reached = table[f'reached_{target}'].astype(str) == 'True'
        table[f't{target}'] = np.where(reached, table[f'time_{target}_capped'], np.nan)
    table['pool_exhausted'] = table.pool_exhausted.astype(str) == 'True'
    metrics = [m for m in ['initial_accuracy', 'auc_accuracy_percent', 'accuracy_300s', 'accuracy_600s', 'accuracy_1200s',
                           'accuracy_2400s', 'acquired_answers', 'acquired_error_rate', 'answers_after_lapse',
                           'pool_exhausted'] if m in table]
    table.groupby(['setting', 'seed_size', 'policy'])[metrics].mean().round(4).to_csv(out / 'stage2_summary.csv')
    budgets = sorted((c for c in table if c.startswith('accuracy_') and c.endswith('s')), key=lambda c: float(c[9:-1]))
    rows = []
    for (setting, size), part in table.groupby(['setting', 'seed_size']):
        for metric in ['auc_accuracy_percent', budgets[-1]]:
            wide = part.pivot(index='seed', columns='policy', values=metric)
            for rival in [p for p in STAGE2_POLICIES if p != 'qmdp' and p in wide]:
                rows.append(test_row((wide['qmdp'] - wide[rival]).to_numpy(), setting=setting, seed_size=size, metric=metric,
                                     rival=rival, primary=PRIMARY_RIVALS.get(setting) == rival))
        rival = PRIMARY_RIVALS.get(setting)
        for target in ['t75', 't80']:
            wide = part.pivot(index='seed', columns='policy', values=target)
            if rival in wide:
                saved = (wide[rival] - wide['qmdp']).to_numpy()      # seconds saved by QMDP; fully observed pairs only
                rows.append(test_row(saved, setting=setting, seed_size=size, metric=f'seconds_saved_{target}', rival=rival,
                                     primary=True))
    tests = pd.DataFrame(rows)
    tests['p_holm'] = tests.groupby(['metric', 'rival']).p.transform(lambda p: holm(p.to_numpy()))
    tests.round(6).to_csv(out / 'stage2_tests.csv', index=False)
    primary = tests[tests.primary & (tests.metric == 'auc_accuracy_percent')].copy()
    primary['p_holm_primary_family'] = holm(primary.p.to_numpy())       # the 12 pre-named primary tests as one family
    primary.round(6).to_csv(out / 'stage2_primary.csv', index=False)
    better = int(((primary.p_holm_primary_family < .05) & (primary.difference > 0)).sum())
    worse = int(((primary.p_holm_primary_family < .05) & (primary.difference < 0)).sum())
    return {'primary_groups': len(primary), 'qmdp_clearly_better_after_holm': better, 'qmdp_clearly_worse_after_holm': worse,
            'runs': len(table), 'final_test_evaluated': bool(table.final_test_evaluated.astype(str).eq('True').all())}


def analyse_grid(out):
    pairs = pd.read_csv(out / 'grid_qmdp_vs_rivals.csv', **READ)
    required = pairs[pairs.second.isin(REQUIRED)]
    beats = required.groupby(['case', 'model', 'beta']).ci_low.min().gt(0).rename('beats').reset_index()
    rates = {m: f'{int(g.sum())}/{len(g)}' for m, g in beats[beats.beta >= 1].groupby('model').beats}
    versus = {}
    for (rival, model), part in pairs[pairs.model != 'None'].groupby(['second', 'model']):
        versus.setdefault(rival, {})[model] = {'clearly_ahead': round(float((part.ci_low > 0).mean()), 3),
                                               'clearly_behind': round(float((part.ci_high < 0).mean()), 3)}
    return {'h2_rate_beta_ge_1_unadjusted': rates, 'qmdp_vs_rivals': versus, 'note': 'descriptive rates; no correction'}


def analyse(out=None):
    out = out or OUT
    report = {}
    if (out / 'stage1_sessions.csv.gz').exists():
        report['stage1'] = analyse_stage1(out)
    if (out / 'stage2_runs.csv').exists():
        report['stage2'] = analyse_stage2(out)
    if (out / 'grid_qmdp_vs_rivals.csv').exists():
        report['grid'] = analyse_grid(out)
    (out / 'final_report.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


# ---------------------------------------------------------------- smoke
def smoke(workers=2, stage2_part=True, policies=POLICIES):
    """Every code path, with VALIDATION annotators, VALIDATION evaluation images and smoke seeds. Not a result."""
    check_frozen()
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    picks = [c for c in study.final_test_cases() if c[0] in ('no_fatigue', 'Markov_beta_0.5', 'Markov_beta_1',
                                                               'Markov_beta_2', 'Markov_beta_3', 'Lapse_seconds_beta_1')]
    run_stage1(out=SMOKE_OUT, split='validation', seed=9000, sessions=40, cases=picks, robust=True, policies=policies)
    if stage2_part:
        import src.eval.stage2 as stage2
        options = {**stage2.default_options(), 'budgets_seconds': [150, 300], 'evaluation': 'SMOKE (validation images)'}
        settings = [stage2.final_test_cases()[i] for i in (1, 5)]       # Markov beta=1 and the seconds-onset Lapse world
        run_stage2(workers=workers, out=SMOKE_OUT, evaluation='validation', annotators='validation', seeds=(9000, 9001),
                   sizes=(200,), settings=settings, policies=['qmdp', 'rest_threshold', 'verify', 'two_choice'], options=options)
    run_grid(workers=workers, out=SMOKE_OUT, split='validation', seed=9000, sessions=40, cases=study.full_cases()[::1953][:4],
             policies=policies)
    analyse(SMOKE_OUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['features', 'settings', 'smoke', 'stage1', 'stage2', 'grid', 'analyse'])
    parser.add_argument('--workers', type=int, default=6)
    arguments = parser.parse_args()
    {'features': run_features, 'settings': write_settings, 'smoke': lambda: smoke(min(arguments.workers, 4)),
     'stage1': run_stage1, 'stage2': lambda: run_stage2(arguments.workers), 'grid': lambda: run_grid(arguments.workers),
     'analyse': analyse}[arguments.part]()
