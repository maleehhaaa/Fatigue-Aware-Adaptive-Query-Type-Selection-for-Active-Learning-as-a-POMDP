"""A frozen-feature linear head with weighted complete/partial labels."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.learn.losses import partial_label_loss


class PictureLearner:
    def __init__(self, data, seed, options):
        torch.manual_seed(seed)
        self.data, self.options = data, options
        self.head = nn.Linear(512, 10)
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=options['initial_lr'], weight_decay=.01)
        self.temperature = 1.
        self.temperature_boundary = False

    def fit(self, image_ids, targets, weights, initial=False):
        rows = self.data['row_for_id'][image_ids]
        assert (rows >= 0).all(), 'A protected or unknown image was requested.'
        features = self.data['features'][rows]
        allowed = torch.tensor(np.array(targets), dtype=torch.bool)
        reliability = torch.tensor(weights, dtype=torch.float32)
        epochs = self.options['initial_epochs'] if initial else self.options['update_epochs']
        for group in self.optimizer.param_groups:
            group['lr'] = self.options['initial_lr'] if initial else self.options['update_lr']
        for _ in range(epochs):
            self.optimizer.zero_grad()
            loss = partial_label_loss(self.head(features), allowed, reliability)
            loss.backward()
            self.optimizer.step()
        self.calibrate()
        return float(loss.detach())

    def scores(self, image_ids):
        rows = self.data['row_for_id'][image_ids]
        assert (rows >= 0).all(), 'A protected or unknown image was requested.'
        with torch.no_grad():
            return self.head(self.data['features'][rows])

    def probabilities(self, image_ids):
        return (self.scores(image_ids) / self.temperature).softmax(1).numpy()

    def calibrate(self):
        scores = self.scores(self.data['calibration_ids'])
        temperatures = torch.logspace(np.log10(.05), np.log10(20), 81)
        scaled = scores[None] / temperatures[:, None, None]
        target = self.data['calibration_labels']
        losses = (scaled.logsumexp(2) - scaled[:, torch.arange(len(target)), target]).mean(1)
        best = int(losses.argmin())
        self.temperature = float(temperatures[best])
        self.temperature_boundary = best in [0, len(temperatures) - 1]

    def accuracy(self):
        predicted = self.scores(self.data['evaluation_ids']).argmax(1)
        return float(predicted.eq(self.data['evaluation_labels']).float().mean()) * 100
