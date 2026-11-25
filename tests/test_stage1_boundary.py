"""Checks that protect the meaning of the experiments, rather than plotting details."""
import ast
import unittest
from pathlib import Path
import numpy as np
from src.common import load_calibration, load_contexts
from src.sim.channel import answer_channel, mutual_information
from src.sim.fatigue import Scenario, transition_table
from src.sim.world import AnnotationWorld
from src.eval.stage1 import population_model, next_belief
from src.control.qmdp import PlanningModel


class ExperimentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_calibration()
        cls.contexts = load_contexts()

    def test_channels_and_information_bound(self):
        p = self.contexts['probabilities'][:20]
        for action in range(5):
            channel = answer_channel(p, np.ones(20) * 3, np.arange(20) % 5,
                action, 2, 0, self.calibration['confusion'])
            self.assertTrue(np.allclose(channel.sum(2), 1))
            info = mutual_information(p, channel)
            entropy = -(p * np.log2(p.clip(1e-12))).sum(1)
            self.assertTrue((info <= entropy + 1e-9).all())

    def test_no_fatigue_means_no_hidden_yes_bias(self):
        world = AnnotationWorld(self.calibration, self.contexts, Scenario(model='None', beta=0), episodes=20)
        for _ in range(20):
            world.step(np.ones(20, dtype=int))
            self.assertTrue((world.oracle_fatigue() == 0).all())

    def test_observations_hide_ordinary_correctness(self):
        world = AnnotationWorld(self.calibration, self.contexts, Scenario(), episodes=10)
        observation = world.step(np.ones(10, dtype=int))
        self.assertEqual(set(observation), {'answer', 'seconds', 'gold', 'gold_correct'})
        self.assertFalse(observation['gold_correct'].any())

    def test_filter_uses_observation_before_transition(self):
        model = PlanningModel(self.contexts, Scenario(), population_model(self.calibration))
        world = AnnotationWorld(self.calibration, self.contexts, Scenario(), episodes=10)
        belief = np.tile([1., 0, 0, 0, 0], (10, 1))
        actions = np.full(10, 3)
        observation = world.step(actions)
        result = next_belief(model, world.indices[0], belief, actions, observation, 0)
        expected = model.transition[world.indices[0], 0, 3]
        self.assertTrue(np.allclose(result, expected))

    def test_controllers_have_no_private_state_access(self):
        forbidden = {'true_labels', 'theta', 'person_ids', 'image_errors', 'load_calibration', 'oracle_fatigue'}
        for path in Path('src/control').glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    self.assertFalse(node.attr.startswith('_'), f'Private access in {path}: {node.attr}')
                if isinstance(node, ast.Name):
                    self.assertNotIn(node.id, forbidden, str(path))
        public = population_model(self.calibration)
        self.assertEqual(set(public), {'skill', 'mu', 'sigma', 'confusion', 'mu_sd', 'sigma_within'})
        # The speed-tracker fields (19 Nov review) must be population facts from training people:
        # changing a validation or test person's own speed must not change them.
        changed = dict(self.calibration)
        changed['mu'] = self.calibration['mu'].copy()
        changed['sigma'] = self.calibration['sigma'].copy()
        held_out = self.calibration['person_split'] != 'train'
        changed['mu'][held_out] += 5
        changed['sigma'][held_out] *= 3
        altered = population_model(changed)
        self.assertEqual(public['mu_sd'], altered['mu_sd'])
        self.assertEqual(public['sigma_within'], altered['sigma_within'])

    def test_transition_rows_are_probabilities(self):
        transitions = transition_table(self.contexts['difficulty'], Scenario())
        self.assertTrue(np.allclose(transitions.sum(3), 1))

    def test_falcon_does_not_recover_on_skips(self):
        transitions = transition_table(self.contexts['difficulty'], Scenario(model='FALCON'))
        self.assertTrue(np.allclose(transitions.sum(3), 1))
        self.assertTrue(np.allclose(transitions[:, :, 0, :], np.eye(5)[None]))

    def test_lapse_process_is_separate_from_fatigue_sensitivity(self):
        # Legacy 'rare' mode: force an onset draw, without a flaky rare-event test.
        world = AnnotationWorld(self.calibration, self.contexts,
            Scenario(model='Lapse', beta=0, lapse_dependence=0, lapse_mode='rare'), episodes=10)
        world._random[0, :, 2] = 0
        world.step(np.ones(10, dtype=int))
        self.assertTrue(world._lapse.all())

    def test_forced_lapse_starts_after_its_scheduled_answer(self):
        # Default 'forced' mode (19 Nov review): one lapse per session, even with beta = 0.
        world = AnnotationWorld(self.calibration, self.contexts,
            Scenario(model='Lapse', beta=0, lapse_onset_min=3., lapse_onset_max=3.), episodes=10)
        for answered in range(1, 5):
            world.step(np.ones(10, dtype=int))
            self.assertEqual(bool(world.lapsed().all()), answered >= 3)
            self.assertFalse(world.lapsed().any() and answered < 3)


if __name__ == '__main__':
    unittest.main()
