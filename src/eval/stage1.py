"""Paired information-level experiments. There is no classifier retraining here."""
import hashlib
import numpy as np
import pandas as pd
from src.common import ROOT, load_calibration, sigmoid, save_run
from src.sim.world import AnnotationWorld
from src.control.belief import observation_likelihood, update_belief, SpeedTracker
from src.control.baselines import baseline_action
from src.control.qmdp import PlanningModel
from src.sim.fatigue import deployable_assumption
from src.control.pomdp_grid import GridPlanner


def population_model(calibration):
    """Population-level facts from training people only; nothing about the person being simulated."""
    train = calibration['person_split'] == 'train'
    return {'skill': float(calibration['public_skill']), 'mu': float(calibration['public_mu']),
        'sigma': float(calibration['public_sigma']), 'confusion': calibration['confusion'].copy(),
        # Spread of personal speeds and within-person noise, for the speed tracker (review fix).
        'mu_sd': float(calibration['mu'][train].std()),
        'sigma_within': float(np.sqrt(np.mean(calibration['sigma'][train] ** 2)))}


def new_speed_tracker(model, count):
    population = model.population
    if 'mu_sd' not in population:
        return None
    return SpeedTracker(count, population['mu_sd'], population['sigma_within'])


def next_belief(model, indices, belief, actions, observations, step, speed=None):
    predicted_d = np.where(actions == 4, 4, model.contexts['difficulty'][indices])
    expected_log_time = (model.population['mu'] + .8 * (1 - sigmoid(predicted_d))
        + np.log(model.scenario.times[actions]) - .06 * step / 100)
    residual = np.log(observations['seconds']) - expected_log_time
    # Review fix: remove the person's own estimated speed before reading fatigue from time.
    offset = speed.offset if speed is not None else 0
    z = (residual - offset) / model.population['sigma']
    accuracy = model.correct[indices, :, actions]
    if speed is not None:
        expected_shift = (model.scenario.signal * model.population['sigma'] * (belief @ np.arange(5))
            + .4 * (1 - (belief * accuracy).sum(1)))
        speed.update(residual, expected_shift, actions > 0)
    likelihood = observation_likelihood(z, accuracy, model.scenario.signal, .4 / model.population['sigma'])
    gold = actions == 4
    if gold.any():
        likelihood[gold] = observation_likelihood(z[gold], accuracy[gold], model.scenario.signal,
            .4 / model.population['sigma'], observations['gold_correct'][gold])
    likelihood[actions == 0] = 1
    transitions = model.transition[indices, :, actions, :]
    return update_belief(belief, likelihood, transitions)


def run_policy(contexts, scenario, name, split='validation', episodes=500, seed=42,
               model=None, grid=None, dqn=None, trace=False, calibration=None, use_speed=True):
    calibration = load_calibration() if calibration is None else calibration
    model = model or PlanningModel(contexts, scenario, population_model(calibration))
    world = AnnotationWorld(calibration, contexts, scenario, split, episodes, seed)
    belief = np.tile([1., 0, 0, 0, 0], (episodes, 1))
    speed = new_speed_tracker(model, episodes) if use_speed else None
    tracking_error = np.zeros(episodes)  # mean |estimated - true fatigue|, evaluation only
    history = np.zeros((episodes, 5), dtype=int)
    random = np.random.default_rng(seed + 100000)
    traces = []
    for step in range(scenario.steps):
        indices = world.indices[step]
        if name == 'oracle':
            # Evaluation-only reference; ordinary controllers never receive this state.
            future = np.einsum('nfag,g->nfa', model.transition[indices], model.value[step + 1])
            values = (model.reward[indices] + future)[:, :, :4]
            fatigue = np.rint(world.oracle_fatigue()).astype(int).clip(0, 4)
            actions = values[np.arange(episodes), fatigue].argmax(1)
        elif name in ['blind', 'myopic', 'qmdp', 'rest_threshold']:
            actions = model.action(indices, belief, step, name)
        elif name == 'grid':
            actions = grid.action(contexts, indices, belief, step)
        elif name == 'dqn':
            from src.control.dqn import encode_state
            import torch
            state = encode_state(contexts, indices, belief, step, scenario.steps, history)
            with torch.no_grad():
                actions = dqn(torch.from_numpy(state)).argmax(1).numpy()
        else:
            actions = baseline_action(name, contexts['probabilities'][indices], belief, step, random)
        tracking_error += np.abs(belief @ np.arange(5) - world.oracle_fatigue().clip(0, 4))
        observations = world.step(actions)
        if trace:
            traces.append(pd.DataFrame({'step': step, 'episode': np.arange(episodes),
                'fatigue': world.last['fatigue'], 'estimated_fatigue': belief @ np.arange(5),
                'action': actions, 'seconds': observations['seconds'], 'correct': world.last['correct']}))
        belief = next_belief(model, indices, belief, actions, observations, step, speed)
        history = np.column_stack([history[:, 1:], actions])
    result = world.results()
    result['tracking_error'] = tracking_error / scenario.steps
    result['lapsed'] = world.lapsed()
    result['trace'] = pd.concat(traces, ignore_index=True) if trace else None
    return result


def benchmark(contexts, scenario, policies, split='validation', episodes=500, seed=42,
              dqn=None, assumed=None, stage='stage1', context_count=16):
    config = {'scenario': scenario.dictionary(), 'assumed': (assumed or scenario).dictionary(),
        'policies': policies, 'split': split, 'episodes': episodes, 'context_count': context_count,
        'context_source': '02: four calibrated checkpoints, top-500 uncertain pool each',
        'code_sha256': hashlib.sha256(b''.join(path.read_bytes() for path in sorted((ROOT / 'src').rglob('*.py')))).hexdigest(),
        'contexts_sha256': hashlib.sha256(contexts['probabilities'].tobytes() + contexts['difficulty'].tobytes()).hexdigest(),
        'calibration_sha256': hashlib.sha256((ROOT / 'data/processed/simulator_calibration.npz').read_bytes()).hexdigest(),
        'dqn_sha256': hashlib.sha256(b''.join(value.detach().numpy().tobytes() for value in dqn.state_dict().values())).hexdigest() if dqn is not None else None,
        'version': 2}
    folder = save_run(stage, config, seed)
    results_path = folder / 'episodes.csv.gz'
    if results_path.exists():
        return pd.read_csv(results_path), folder
    calibration = load_calibration()
    population = population_model(calibration)
    model = PlanningModel(contexts, assumed or scenario, population)
    grid = GridPlanner(model, context_count=context_count) if 'grid' in policies else None
    # Review fix: a "deployable" QMDP that plans with fixed default assumptions in every world.
    fixed = PlanningModel(contexts, deployable_assumption(scenario), population) if 'qmdp_fixed' in policies else None
    if 'rest_threshold' in policies:
        model.rest_threshold = tune_rest_threshold(contexts, scenario, model, calibration, seed=seed)
    rows = []
    for name in policies:
        if name == 'qmdp_fixed':
            result = run_policy(contexts, scenario, 'qmdp', split, episodes, seed, fixed, calibration=calibration)
        else:
            result = run_policy(contexts, scenario, name, split, episodes, seed, model, grid, dqn, calibration=calibration)
        table = pd.DataFrame({key: result[key] for key in ['information', 'seconds', 'reward', 'error_rate',
            'questions', 'tracking_error', 'lapsed']})
        table['episode'], table['policy'] = np.arange(episodes), name
        for action, action_name in enumerate(['skip', 'verify', 'two_choice', 'full', 'gold']):
            table[f'count_{action_name}'] = result['action_counts'][:, action]
        rows.append(table)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(results_path, index=False)
    return table, folder


def tune_rest_threshold(contexts, scenario, model, calibration, thresholds=(1., 1.5, 2., 2.5, 3.),
                        episodes=200, seed=42):
    """Pick the greedy-plus-rest threshold on training people only (never validation)."""
    scores = {}
    for threshold in thresholds:
        model.rest_threshold = threshold
        result = run_policy(contexts, scenario, 'rest_threshold', 'train', episodes, seed + 500,
            model, calibration=calibration)
        scores[threshold] = result['reward'].mean()
    return max(scores, key=scores.get)


def summarize(table):
    columns = ['reward', 'information', 'seconds', 'error_rate'] + [c for c in ['tracking_error'] if c in table]
    summary = table.groupby('policy')[columns].mean()
    summary['bits_per_second'] = summary.information / summary.seconds
    summary['reward_se'] = table.groupby('policy').reward.sem()
    return summary.sort_values('reward', ascending=False)
