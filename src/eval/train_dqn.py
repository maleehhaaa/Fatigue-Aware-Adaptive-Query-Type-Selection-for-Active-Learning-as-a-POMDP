"""Train a policy in the simulator; private evaluation code, outside controllers."""
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from src.control.dqn import encode_state, make_network


def train_dqn(contexts, calibration, population, rounds=80, parallel=32, seed=42,
              updates_per_step=4, discount=.99, reward_scale=.1, target_mix=.01):
    """Double DQN on simulated training people.

    Review fix: the old loop made one gradient step every 4 environment steps (about 3,000 in
    total) with ~30 hard target copies, too few to pass value back over a 150-step undiscounted
    horizon. Now: several steps per environment step (80 rounds x 150 x 4 = 48,000), discount
    0.99, scaled rewards and a soft target update.
    """
    # Imports stay here so policy inference never needs the simulator.
    from src.sim.world import AnnotationWorld
    from src.sim.fatigue import Scenario
    from src.control.qmdp import PlanningModel
    from src.eval.stage1 import next_belief, new_speed_tracker
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    random = np.random.default_rng(seed)
    network = make_network()
    target = copy.deepcopy(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    capacity, position, available, updates = 60000, 0, 0, 0
    states = np.zeros((capacity, 16), dtype='float32')
    next_states = np.zeros_like(states)
    actions_buffer = np.zeros(capacity, dtype='int64')
    rewards = np.zeros(capacity, dtype='float32')
    terminal = np.zeros(capacity, dtype='float32')
    logs, models = [], {}
    for round_number in range(rounds):
        beta = float(random.choice([.5, 1., 2., 3.]))
        model_name = str(random.choice(['Markov', 'Markov', 'Markov', 'FALCON', 'Lapse', 'None']))
        distribution = str(random.choice(['constant', 'uniform', 'mixture', 'gamma']))
        signal, recovery = float(random.choice([.5, 1., 2.])), float(random.choice([.2, .35, .5]))
        scenario = Scenario(model=model_name, beta=0 if model_name == 'None' else beta,
            distribution=distribution, signal=signal, recovery=recovery)
        key = (model_name, scenario.beta, signal, recovery)
        if key not in models:
            models[key] = PlanningModel(contexts, scenario, population)
        model = models[key]
        world = AnnotationWorld(calibration, contexts, scenario, 'train', parallel, seed + round_number)
        belief = np.tile([1., 0, 0, 0, 0], (parallel, 1))
        speed = new_speed_tracker(model, parallel)
        history = np.zeros((parallel, 5), dtype=int)
        epsilon = max(.05, 1 - round_number / (rounds * .75))
        round_losses = []
        for step in range(scenario.steps):
            state = encode_state(contexts, world.indices[step], belief, step, scenario.steps, history)
            with torch.no_grad():
                actions = network(torch.from_numpy(state)).argmax(1).numpy()
            explore = random.random(parallel) < epsilon
            actions[explore] = random.integers(0, 5, explore.sum())
            observations = world.step(actions)
            updated = next_belief(model, world.indices[step], belief, actions, observations, step, speed)
            history = np.column_stack([history[:, 1:], actions])
            future_state = encode_state(contexts, world.indices[step + 1], updated, step + 1, scenario.steps, history)
            slots = (np.arange(parallel) + position) % capacity
            states[slots], next_states[slots] = state, future_state
            actions_buffer[slots], rewards[slots] = actions, world.last['reward'] * reward_scale
            terminal[slots] = step == scenario.steps - 1
            position, available = (position + parallel) % capacity, min(capacity, available + parallel)
            belief = updated
            if available < 1024:
                continue
            for _ in range(updates_per_step):
                sampled = random.integers(available, size=256)
                current = network(torch.from_numpy(states[sampled])).gather(1, torch.tensor(actions_buffer[sampled, None])).squeeze(1)
                with torch.no_grad():
                    future = torch.from_numpy(next_states[sampled])
                    best = network(future).argmax(1, keepdim=True)
                    next_value = target(future).gather(1, best).squeeze(1)
                    expected = torch.from_numpy(rewards[sampled]) + discount * (1 - torch.from_numpy(terminal[sampled])) * next_value
                loss = F.smooth_l1_loss(current, expected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 10)
                optimizer.step()
                updates += 1
                with torch.no_grad():
                    for target_value, value in zip(target.parameters(), network.parameters()):
                        target_value.mul_(1 - target_mix).add_(value, alpha=target_mix)
            round_losses.append(loss.item())
        logs.append({'round': round_number + 1, 'model': model_name, 'beta': scenario.beta,
            'epsilon': epsilon, 'mean_reward': world.results()['reward'].mean(),
            'loss': np.mean(round_losses) if round_losses else np.nan, 'updates': updates})
        if (round_number + 1) % 10 == 0:
            print(f'DQN practice round {round_number + 1}/{rounds}', flush=True)
    network.eval()
    return network, pd.DataFrame(logs)
