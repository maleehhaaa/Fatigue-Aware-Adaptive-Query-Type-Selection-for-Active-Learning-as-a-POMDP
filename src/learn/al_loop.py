"""Real classifier learning from public human answers, with a seconds budget."""
import numpy as np
import pandas as pd
from src.learn.classifier import PictureLearner
from src.learn.losses import answer_target
from src.control.qmdp import PlanningModel
from src.control.baselines import baseline_action
from src.eval.stage1 import next_belief
from src.control.belief import SpeedTracker
from src.sim.fatigue import deployable_assumption

MODEL_POLICIES = {'qmdp': 'qmdp', 'qmdp_fixed': 'qmdp', 'myopic': 'myopic', 'blind': 'blind',
                  'rest_threshold': 'rest_threshold'}


def run_learning(data, seed_ids, seed_labels, scenario, policy, seed, options, population, session_factory):
    learner = PictureLearner(data, seed, options)
    training_ids = list(seed_ids)
    targets = list(np.eye(10, dtype=bool)[seed_labels])
    weights = [1.] * len(seed_ids)
    loss = learner.fit(training_ids, targets, weights, initial=True)
    accuracy = learner.accuracy()
    curve = [{'seconds': 0., 'accuracy_percent': accuracy, 'targets_used': len(training_ids),
              'temperature': learner.temperature, 'loss': loss}]
    available = np.zeros(len(data['row_for_id']), dtype=bool)
    available[data['pool_ids']] = True
    available[seed_ids] = False
    clock, session, asked, practice, call = 0., -1, 0, 0, 0
    events = []
    refresh = True
    action_random = np.random.default_rng(seed + 110000)
    budget, shift = max(options['budgets_seconds']), options['session_seconds']
    skip_seconds = scenario.pause_duration(population['mu'])
    # Review fix: a deployable QMDP plans with fixed default assumptions, not the true world's.
    planning_scenario = deployable_assumption(scenario) if policy == 'qmdp_fixed' else scenario
    while clock < budget - 1e-9 and available.any():
        current_session = int((clock + 1e-9) // shift)
        if current_session != session:
            session = current_session
            ask = session_factory(session)
            belief = np.array([[1., 0, 0, 0, 0]])
            # Each shift is a new person, so their speed is learned again from scratch.
            speed = (SpeedTracker(1, population['mu_sd'], population['sigma_within'])
                     if 'mu_sd' in population else None)
            practice, opportunities = 0, 0
            candidate_random = np.random.default_rng(seed * 10000 + session + 220000)
        if refresh:
            remaining_ids = np.flatnonzero(available)
            probabilities = learner.probabilities(remaining_ids)
            entropy = -(probabilities * np.log2(probabilities.clip(1e-12))).sum(1)
            chosen = np.arange(len(remaining_ids)) if policy == 'random_image_full' else np.argsort(entropy)[-500:]
            ids = remaining_ids[chosen]
            contexts = {'image_id': ids, 'probabilities': probabilities[chosen], 'difficulty': data['predicted_d'][ids]}
            planning_indices = np.linspace(0, len(ids) - 1, min(64, len(ids))).astype(int)
            model = PlanningModel(contexts, planning_scenario, population,
                plan=policy in ['qmdp', 'qmdp_fixed'], planning_indices=planning_indices)
            model.rest_threshold = options.get('rest_threshold', 2.0)
            refresh = False
        eligible = np.flatnonzero(available[contexts['image_id']])
        index = eligible[min(int(candidate_random.random() * len(eligible)), len(eligible) - 1)]
        remaining_time = min((session + 1) * shift, budget) - clock
        # Receding look-ahead: at most 150 opportunities, shortened near a hard boundary.
        horizon = min(scenario.steps, max(1, int(remaining_time / skip_seconds)))
        estimated_fatigue = float(belief[0] @ np.arange(5))
        if policy in MODEL_POLICIES:
            action = int(model.action(np.array([index]), belief, scenario.steps - horizon, MODEL_POLICIES[policy])[0])
        else:
            name = 'full' if policy == 'random_image_full' else policy
            action = int(baseline_action(name, contexts['probabilities'][index:index + 1], belief, opportunities, action_random)[0])
        reliability = float(belief[0] @ model.correct[index, :, action])
        observation = ask(contexts, index, action)
        duration = float(observation['seconds'][0])
        accepted = duration <= remaining_time + 1e-9
        start = clock
        clock = min(clock + duration, (session + 1) * shift, budget)
        top_two = np.argsort(-contexts['probabilities'][index])[:2]
        image_id = int(contexts['image_id'][index])
        answer = int(observation['answer'][0]) if accepted else -1
        events.append({'call': call, 'session': session, 'start_seconds': start, 'seconds': clock,
            'action': action, 'image_id': image_id, 'answer': answer, 'accepted': accepted,
            'estimated_reliability': reliability, 'estimated_fatigue': estimated_fatigue,
            'top_one': int(top_two[0]), 'top_two': int(top_two[1])})
        call += 1
        if not accepted:
            continue  # A response crossing the shift boundary is censored, not used for learning.
        belief = next_belief(model, np.array([index]), belief, np.array([action]), observation, min(practice, 200), speed)
        practice += action > 0
        opportunities += 1
        if action in [1, 2, 3]:
            training_ids.append(image_id)
            targets.append(answer_target(action, answer, top_two))
            weights.append(reliability)
            available[image_id] = False
            asked += 1
            if asked % options['retrain_every_answers'] == 0:
                loss = learner.fit(training_ids, targets, weights)
                accuracy = learner.accuracy()
                curve.append({'seconds': clock, 'accuracy_percent': accuracy, 'targets_used': len(training_ids),
                    'temperature': learner.temperature, 'loss': loss})
                refresh = True
    # Pending answers remain buffered until the next group of ten, as in the specified online learner.
    return {'curve': pd.DataFrame(curve), 'events': pd.DataFrame(events), 'learner': learner,
        'training_ids': np.array(training_ids), 'targets': np.array(targets), 'weights': np.array(weights),
        'stop_seconds': clock, 'pool_exhausted': not available.any(), 'pending_answers': asked % options['retrain_every_answers']}
