"""Learn from an answer's allowed classes, including partial answers."""
import numpy as np
import torch


def answer_target(action, answer, top_two):
    allowed = np.ones(10, dtype=bool)
    if action == 3:
        allowed[:] = False
        allowed[answer] = True
    elif action == 1 and answer == 0:
        allowed[:] = False
        allowed[top_two[0]] = True
    elif action == 1:
        allowed[top_two[0]] = False
    elif action == 2 and answer < 2:
        allowed[:] = False
        allowed[top_two[answer]] = True
    elif action == 2:
        allowed[top_two] = False
    else:
        raise ValueError('Skip and gold do not provide a training target.')
    return allowed


def partial_label_loss(scores, allowed, weights):
    # A complete label allows one class; "no" and "neither" allow several.
    log_mass = torch.logsumexp(scores.masked_fill(~allowed, -torch.inf), dim=1)
    losses = torch.logsumexp(scores, dim=1) - log_mass
    return (weights * losses).sum() / weights.sum().clamp_min(1e-8)
