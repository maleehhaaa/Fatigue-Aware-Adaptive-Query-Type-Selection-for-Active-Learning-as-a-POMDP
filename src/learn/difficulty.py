"""Predict human easiness from image features, using only separate training images."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from src.common import ROOT


def train_difficulty_predictor(calibration, seed=42, loss='mse'):
    """loss='mse' fits logit(p) directly (original). loss='bce' fits the success probability p with
    cross-entropy, which is less dominated by the 35.6% of images at the ceiling (review item 15)."""
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    images = pd.read_csv(ROOT / 'data/splits/images.csv').set_index('image_id')
    with np.load(ROOT / 'data/features/resnet18_development_features.npz') as saved:
        ids = saved['image_ids']
        features = F.normalize(torch.tensor(saved['features']), dim=1)
    train_mask = images.loc[ids, 'image_split'].eq('difficulty_train').to_numpy(copy=True)
    target = torch.tensor(calibration['difficulty'][ids], dtype=torch.float32)
    predictor = nn.Sequential(nn.Linear(512, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=.01, weight_decay=.05)
    losses = []
    for _ in range(300):
        optimizer.zero_grad()
        output = predictor(features[train_mask]).squeeze(1)
        if loss == 'bce':
            value = F.binary_cross_entropy_with_logits(output, torch.sigmoid(target[train_mask]))
        else:
            value = F.mse_loss(output, target[train_mask])
        value.backward()
        optimizer.step()
        losses.append(value.item())
    predictor.eval()
    with torch.no_grad():
        predictions = predictor(features).squeeze(1).numpy().clip(-4, 6)
    table = pd.DataFrame({'image_id': ids, 'predicted_d': predictions})
    table.to_csv(ROOT / 'data/processed/predicted_difficulty.csv', index=False)
    torch.save(predictor.state_dict(), ROOT / 'data/features/difficulty_predictor.pt')
    return table, losses
