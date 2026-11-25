"""Double DQN: learn action scores from observable belief states."""
import numpy as np
import torch
from torch import nn


def encode_state(contexts, indices, belief, step, steps, history):
    probabilities = contexts['probabilities'][indices]
    top = np.sort(probabilities, axis=1)[:, -2:]
    entropy = -(probabilities * np.log2(probabilities.clip(1e-12))).sum(1) / np.log2(10)
    # Stage 1 has a time penalty but no hard time budget; budget fraction is 1.
    return np.column_stack([belief, top[:, ::-1], entropy, contexts['difficulty'][indices] / 4,
        np.full(len(indices), (steps - step) / steps), np.ones(len(indices)), history / 4]).astype('float32')


def make_network():
    return nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 5))


