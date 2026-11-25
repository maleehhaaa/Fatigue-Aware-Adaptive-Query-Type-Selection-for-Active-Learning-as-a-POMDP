"""E3: where does QMDP's advantage come from? Pausing vs question-format choice.

QMDP verifies on about 98% of turns at lambda = 1.0, yet beats always-verify in Markov worlds.
The other 1-2% of actions are pauses (about 5% of the time budget) and a few full / two-choice
questions. These ablations separate the two. Every variant uses QMDP's own planning model
(src/control/qmdp.py); only the set of actions it may plan over and use changes.

  Variant                  Plans over          Uses
  qmdp (reference)         pause, V, 2C, full  same
  qmdp_no_pause            V, 2C, full         same (never pauses)
  qmdp_verify_pause        pause, V            same (never 2C or full)
  verify_with_qmdp_pause   pause, V, 2C, full  pauses when full QMDP would pause, else verifies
  verify (reference)       -                   always verify

Read the chain  verify -> verify_with_qmdp_pause -> qmdp_verify_pause -> qmdp:
  step 1 = what QMDP's pauses add to always-verify,
  step 2 = re-planning the pauses for a verify-only controller,
  step 3 = what format choice adds on top.
And qmdp_no_pause vs qmdp = what pausing adds inside QMDP.

Parts:
  stage1  Information minus time (Stage 1), validation people, 500 matched sessions, plus greedy
          and tuned greedy-plus-rest. Fast.
  stage2  Classifier learning (Stage 2), same design as the development study (validation
          annotators, seeds 42-51, 50 and 200 starting labels, the five settings). The qmdp and
          verify runs are reused from the run cache when it still holds the development study.
  analyse Paired differences (bootstrap 95% CIs, 4,000 resamples) and the decomposition table.
  smoke   Tiny version of every part, separate folder. Not a result.

    python scripts/run_e3_ablation.py stage1
    python scripts/run_e3_ablation.py stage2 --workers 13
    python scripts/run_e3_ablation.py analyse
    python scripts/run_e3_ablation.py smoke

src/ is not changed: the variants are a subclass of PlanningModel defined here. Results go to
outputs/e3_ablation/.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import argparse
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
from src.sim.fatigue import Scenario

OUT = project / 'outputs/e3_ablation'

PAUSE, VERIFY, TWO, FULL = 0, 1, 2, 3
VARIANTS = {'qmdp_no_pause': (VERIFY, TWO, FULL),
            'qmdp_verify_pause': (PAUSE, VERIFY),
            'verify_with_qmdp_pause': (PAUSE, VERIFY, TWO, FULL)}
STAGE1_CASES = {'no_fatigue': Scenario(model='None', beta=0),
                **{f'Markov_beta_{b:g}': Scenario(beta=b) for b in (.5, 1, 2, 3)},
                'FALCON_beta_1': Scenario(model='FALCON', beta=1),
                'Lapse_beta_1': Scenario(model='Lapse', beta=1)}
STAGE1_POLICIES = ['qmdp', *VARIANTS, 'verify', 'myopic', 'rest_threshold']
STAGE2_POLICIES = ['qmdp', *VARIANTS, 'verify']
CHAIN = [('verify', 'verify_with_qmdp_pause', 'pauses added to always-verify'),
         ('verify_with_qmdp_pause', 'qmdp_verify_pause', 'pauses re-planned for verify-only'),
         ('qmdp_verify_pause', 'qmdp', 'format choice added'),
         ('qmdp_no_pause', 'qmdp', 'pausing added inside QMDP'),
         ('verify', 'qmdp_no_pause', 'format choice without pausing'),
         ('verify', 'qmdp', 'total')]


class AblationModel(PlanningModel):
    """QMDP restricted to a set of actions, both when planning and when acting."""

    def __init__(self, contexts, scenario, population, plan=True, planning_indices=None, variant=None):
        super().__init__(contexts, scenario, population, plan=False, planning_indices=planning_indices)
        self.variant = variant
        self.mask = np.full(4, -np.inf)
        self.mask[list(VARIANTS[variant])] = 0.
        reward = self.reward if planning_indices is None else self.reward[planning_indices]
        transition = self.transition if planning_indices is None else self.transition[planning_indices]
        for step in range(scenario.steps - 1, -1, -1):
            future = np.einsum('cfag,g->cfa', transition, self.value[step + 1])
            self.value[step] = (reward[:, :, :4] + future[:, :, :4] + self.mask).max(2).mean(0)

    def action(self, indices, belief, step, name='qmdp'):
        future = np.einsum('nfag,g->nfa', self.transition[indices], self.value[step + 1])
        values = np.einsum('nf,nfa->na', belief, (self.reward[indices] + future)[:, :, :4])
        if self.variant == 'verify_with_qmdp_pause':
            return np.where(values.argmax(1) == PAUSE, PAUSE, VERIFY)
        return (values + self.mask).argmax(1)


def bootstrap(values, seed=42, resamples=4000):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    means = values[np.random.default_rng(seed).integers(len(values), size=(resamples, len(values)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return values.mean(), low, high


# ------------------------------------------------------------------ Stage 1
def run_stage1(cases=STAGE1_CASES, episodes=500, seed=42):
    OUT.mkdir(parents=True, exist_ok=True)
    contexts, calibration = load_contexts(), load_calibration()
    population = stage1.population_model(calibration)
    rows = []
    for case, scenario in cases.items():
        model = PlanningModel(contexts, scenario, population)
        model.rest_threshold = stage1.tune_rest_threshold(contexts, scenario, model, calibration, seed=seed)
        for policy in STAGE1_POLICIES:
            use = AblationModel(contexts, scenario, population, variant=policy) if policy in VARIANTS else model
            name = 'qmdp' if policy in VARIANTS else policy
            result = stage1.run_policy(contexts, scenario, name, 'validation', episodes, seed, use, calibration=calibration)
            table = pd.DataFrame({key: result[key] for key in ['reward', 'information', 'seconds', 'error_rate',
                                                               'tracking_error']})
            for action, label in enumerate(['pause', 'verify', 'two_choice', 'full', 'gold']):
                table[f'count_{label}'] = result['action_counts'][:, action]
            table['episode'], table['policy'], table['case'] = np.arange(episodes), policy, case
            table['rest_threshold'] = model.rest_threshold
            rows.append(table)
        print(f'{case}: done', flush=True)
    episodes_table = pd.concat(rows, ignore_index=True)
    episodes_table.to_csv(OUT / 'stage1_episodes.csv.gz', index=False)
    summary = episodes_table.groupby(['case', 'policy']).mean(numeric_only=True).drop(columns='episode')
    summary['bits_per_second'] = summary.information / summary.seconds
    summary.round(4).to_csv(OUT / 'stage1_summary.csv')
    paired = []
    for case, part in episodes_table.groupby('case'):
        wide = part.pivot(index='episode', columns='policy', values='reward')
        for first, second, label in CHAIN + [('myopic', 'qmdp', 'QMDP vs greedy'), ('rest_threshold', 'qmdp', 'QMDP vs greedy-plus-rest')]:
            mean, low, high = bootstrap(wide[second] - wide[first])
            paired.append({'case': case, 'step': label, 'from': first, 'to': second,
                           'reward_difference': mean, 'ci_low': low, 'ci_high': high})
    pd.DataFrame(paired).round(3).to_csv(OUT / 'stage1_decomposition.csv', index=False)
    print(pd.DataFrame(paired).round(2).to_string(index=False))


# ------------------------------------------------------------------ Stage 2
CURRENT = {'policy': None}


def model_factory(contexts, scenario, population, plan=True, planning_indices=None):
    """Replaces al_loop.PlanningModel inside the workers. Variants get the restricted planner."""
    variant = CURRENT['policy']
    if variant in VARIANTS:
        return AblationModel(contexts, scenario, population, planning_indices=planning_indices, variant=variant)
    return PlanningModel(contexts, scenario, population, plan=plan, planning_indices=planning_indices)


def initialize_ablation_worker():
    import src.eval.stage2 as stage2
    import src.learn.al_loop as al_loop
    stage2.initialize_worker()
    al_loop.PlanningModel = model_factory
    for variant in VARIANTS:
        al_loop.MODEL_POLICIES[variant] = 'qmdp'   # routed through model.action (the variant decides)


def ablation_job(job):
    import src.eval.stage2 as stage2
    CURRENT['policy'] = job[1]
    return stage2.learning_job(job)


def run_stage2(workers=6, seeds=range(42, 52), seed_sizes=(50, 200), options=None, cases=None, tag='stage2_runs'):
    import src.eval.stage2 as stage2
    OUT.mkdir(parents=True, exist_ok=True)
    options = options or stage2.default_options()
    cases = cases or stage2.experiment_cases()
    jobs = [(scenario, policy, size, seed, options) for scenario in cases for size in seed_sizes
            for seed in seeds for policy in STAGE2_POLICIES]
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    print(f'{len(jobs)} classifier runs on {workers} workers (qmdp and verify come from the cache if present)...', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_ablation_worker) as pool:
        futures = [pool.submit(ablation_job, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            summary = future.result()
            summary['run_folder'] = Path(summary['run_folder']).as_posix()
            rows.append(summary)
            if count % 20 == 0 or count == len(jobs):
                print(f'  {count}/{len(jobs)} done', flush=True)
    table = pd.DataFrame(rows).sort_values(['model', 'beta', 'seed_size', 'seed', 'policy'])
    table.to_csv(OUT / f'{tag}.csv', index=False)
    return table


def audit_measures(run_folder):
    audit = pd.read_csv(project / run_folder / 'answer_audit.csv.gz', keep_default_na=False, na_values=[''])
    accepted = audit[audit.accepted.astype(str) == 'True'].copy()
    accepted['duration'] = accepted.seconds - accepted.start_seconds
    asked = accepted[accepted.action.isin([1, 2, 3])]
    verify = asked[asked.action == 1]
    return {'pauses': int((accepted.action == 0).sum()), 'pause_seconds': float(accepted.loc[accepted.action == 0, 'duration'].sum()),
            'non_verify_questions': int(asked.action.isin([2, 3]).sum()),
            'true_fatigue_when_asking': float(asked.fatigue.mean()),
            'seconds_per_verify': float(verify.duration.mean()) if len(verify) else np.nan,
            'verify_correct': float((verify.correct.astype(str) == 'True').mean()) if len(verify) else np.nan}


def analyse(tag='stage2_runs'):
    table = pd.read_csv(OUT / f'{tag}.csv', keep_default_na=False, na_values=[''])
    extra = pd.DataFrame([audit_measures(folder) for folder in table.run_folder])
    table = pd.concat([table.reset_index(drop=True), extra], axis=1)
    table.to_csv(OUT / f'{tag}_with_behaviour.csv', index=False)
    # Accuracy at each budget that exists (the smoke test uses shorter budgets than the study).
    budgets = sorted((c for c in table if c.startswith('accuracy_') and c.endswith('s')), key=lambda c: float(c[9:-1]))
    columns = [c for c in ['auc_accuracy_percent', *budgets, 'acquired_answers', 'acquired_error_rate',
               'answers_after_lapse'] if c in table] + list(extra.columns)
    table.groupby(['model', 'beta', 'seed_size', 'policy'])[columns].mean().round(3).to_csv(OUT / f'{tag}_summary.csv')
    paired = []
    for keys, part in table.groupby(['model', 'beta', 'seed_size']):
        # Average accuracy, the earliest and the last budget, and answers collected.
        for metric in dict.fromkeys(['auc_accuracy_percent', budgets[0], budgets[-1], 'acquired_answers']):
            wide = part.pivot(index='seed', columns='policy', values=metric)
            for first, second, label in CHAIN:
                mean, low, high = bootstrap(wide[second] - wide[first])
                paired.append({'model': keys[0], 'beta': keys[1], 'seed_size': keys[2], 'metric': metric, 'step': label,
                               'from': first, 'to': second, 'difference': mean, 'ci_low': low, 'ci_high': high})
    paired = pd.DataFrame(paired).round(3)
    paired.to_csv(OUT / f'{tag}_decomposition.csv', index=False)
    print(paired[paired.metric == 'auc_accuracy_percent'].to_string(index=False))


# ------------------------------------------------------------------ smoke
def smoke():
    """Tiny version of every part, written to a separate folder. Not a result."""
    global OUT
    OUT = project / 'outputs/e3_ablation_smoke'
    run_stage1(cases={'Markov_beta_1': Scenario(beta=1)}, episodes=40)
    import src.eval.stage2 as stage2
    options = {**stage2.default_options(), 'budgets_seconds': [150, 300]}
    run_stage2(workers=2, seeds=(42,), seed_sizes=(200,), options=options, cases=[Scenario(beta=1)])
    analyse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['stage1', 'stage2', 'analyse', 'smoke'])
    parser.add_argument('--workers', type=int, default=6)
    arguments = parser.parse_args()
    if arguments.part == 'stage1':
        run_stage1()
    elif arguments.part == 'stage2':
        run_stage2(arguments.workers)
    elif arguments.part == 'analyse':
        analyse()
    else:
        smoke()
