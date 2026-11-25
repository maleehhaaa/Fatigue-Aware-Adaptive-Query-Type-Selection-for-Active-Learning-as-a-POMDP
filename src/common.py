"""Small file helpers shared by the experiment notebooks."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def save_run(stage, config, seed=42):
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    folder = ROOT / 'outputs' / stage / key / f'seed-{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    return folder


def load_calibration():
    return dict(np.load(ROOT / 'data/processed/simulator_calibration.npz'))


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def load_contexts():
    """Public model predictions; no true labels or private difficulty."""
    settings = json.loads((ROOT / 'configs/classifier_settings.json').read_text())
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    folder = ROOT / 'outputs/classifier_development' / digest / f"seed-{settings['seed']}"
    difficulty = pd.read_csv(ROOT / 'data/processed/predicted_difficulty.csv').set_index('image_id')
    rows = []
    for count in settings['label_counts']:
        with np.load(folder / f'pool_confidence_{count}_labels.npz') as snapshot:
            probabilities = snapshot['probabilities']
            entropy = -(probabilities * np.log2(probabilities.clip(1e-12))).sum(1)
            # Follow the brief: sample from the 500 most uncertain remaining pictures.
            selected = np.argsort(entropy)[-500:]
            ids = snapshot['image_ids'][selected]
            rows.append((ids, probabilities[selected], difficulty.loc[ids, 'predicted_d'].to_numpy()))
    return {'image_id': np.concatenate([r[0] for r in rows]),
            'probabilities': np.concatenate([r[1] for r in rows]),
            'difficulty': np.concatenate([r[2] for r in rows]),
            'checkpoint': np.repeat(settings['label_counts'], 500)}
