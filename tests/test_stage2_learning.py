"""Protect partial-label learning, time accounting, and the truth boundary."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from src.learn.losses import answer_target, partial_label_loss
from src.learn.al_loop import run_learning
from src.learn.classifier import PictureLearner
from src.eval.stage2 import default_options, sample_curve, curve_auc
from src.sim.fatigue import Scenario


class LearningChecks(unittest.TestCase):
    def test_complete_label_is_cross_entropy(self):
        scores = torch.tensor([[1., 2., 3.]])
        allowed = torch.tensor([[False, True, False]])
        actual = partial_label_loss(scores, allowed, torch.ones(1))
        self.assertTrue(torch.allclose(actual, F.cross_entropy(scores, torch.tensor([1]))))

    def test_negative_answer_pushes_excluded_probability_down(self):
        scores = torch.zeros((1, 10), requires_grad=True)
        allowed = torch.tensor(answer_target(1, 1, np.array([3, 5]))[None])
        loss = partial_label_loss(scores, allowed, torch.ones(1))
        loss.backward()
        self.assertGreater(scores.grad[0, 3], 0)
        self.assertLess(scores.grad[0, 5], 0)

    def test_neither_excludes_both_proposed_classes(self):
        target = answer_target(2, 2, np.array([3, 5]))
        self.assertEqual(target.sum(), 8)
        self.assertFalse(target[3] or target[5])

    def test_zero_weight_wrong_answer_has_no_gradient(self):
        scores = torch.zeros((2, 10), requires_grad=True)
        targets = torch.eye(10, dtype=torch.bool)[[2, 8]]
        partial_label_loss(scores, targets, torch.tensor([1., 0.])).backward()
        self.assertTrue(torch.equal(scores.grad[1], torch.zeros(10)))

    def test_no_future_accuracy_at_budget(self):
        curve = pd.DataFrame({'seconds': [0.,301.], 'accuracy_percent': [50.,90.]})
        self.assertEqual(sample_curve(curve, [300])[0], 50)
        self.assertAlmostEqual(curve_auc(curve, 600), (301*50+299*90)/600)

    def test_censored_answer_never_enters_training(self):
        options = default_options()
        options.update({'budgets_seconds': [.1], 'initial_epochs': 1, 'update_epochs': 1})
        data = {'features': torch.zeros((4,512)), 'row_for_id': np.arange(4),
            'pool_ids': np.array([0,1]), 'predicted_d': np.ones(4)*3,
            'calibration_ids': np.array([2]), 'evaluation_ids': np.array([3]),
            'calibration_labels': torch.tensor([0]), 'evaluation_labels': torch.tensor([0])}
        def factory(session):
            return lambda contexts, index, action: {'seconds': np.array([10.]), 'answer': np.array([9]),
                'gold': np.array([False]), 'gold_correct': np.array([False])}
        population = {'mu': .4, 'sigma': .4, 'skill': 0., 'confusion': (np.ones((10,10))-np.eye(10))/9}
        result = run_learning(data, np.array([0]), np.array([0]), Scenario(), 'full', 42, options, population, factory)
        self.assertEqual(result['training_ids'].tolist(), [0])
        self.assertFalse(result['events'].accepted.any())
        self.assertEqual(result['stop_seconds'], .1)

    def test_learner_never_reads_unknown_pool_truth(self):
        for filename in ['al_loop.py','classifier.py','losses.py']:
            source = (Path('src/learn') / filename).read_text()
            for forbidden in ['true_labels', 'oracle_fatigue', 'load_calibration', 'world.last']:
                self.assertNotIn(forbidden, source)

    def test_protected_image_id_cannot_index_last_feature_row(self):
        data = {'features': torch.zeros((2,512)), 'row_for_id': np.array([0,1,-1])}
        learner = PictureLearner(data, 42, default_options())
        with self.assertRaises(AssertionError):
            learner.scores(np.array([2]))


if __name__ == '__main__':
    unittest.main()
