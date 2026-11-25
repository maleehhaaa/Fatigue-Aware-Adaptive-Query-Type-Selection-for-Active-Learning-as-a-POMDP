"""Protect the fixes from the 19 Nov 2026 code review (time scale, rest, FALCON, lapse, baselines)."""
import unittest
import numpy as np
import pandas as pd
from src.common import load_calibration, load_contexts
from src.sim.fatigue import Scenario, fast_stress_test, deployable_assumption, transition_table
from src.sim.world import AnnotationWorld
from src.control.belief import SpeedTracker
from src.control.qmdp import PlanningModel
from src.eval.stage1 import population_model
from src.eval.stage2 import comparison_tables


class ReviewFixChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_calibration()
        cls.contexts = load_contexts()

    def fatigue_path(self, scenario, action, steps=150, episodes=500):
        world = AnnotationWorld(self.calibration, self.contexts, scenario, episodes=episodes, seed=1)
        path = []
        for _ in range(steps):
            path.append(world.oracle_fatigue())
            world.step(np.full(episodes, action))
        return world, np.array(path)

    def test_fatigue_builds_over_a_session_not_seconds(self):
        _, realistic = self.fatigue_path(Scenario(beta=2), 3)
        _, fast = self.fatigue_path(fast_stress_test(Scenario(beta=2)), 3)
        self.assertLess(realistic[49].mean(), 2)  # not exhausted after 50 full labels
        self.assertGreater(fast[20].mean(), 3)    # the stress preset keeps the old behaviour

    def test_every_scenario_setting_fits_one_table_column(self):
        # run_study copies each setting into a column; a tuple or list breaks it (19 Nov bug).
        from src.eval.study import initial_cases
        from src.eval.stage2 import experiment_cases
        scenarios = [scenario for _, scenario, _ in initial_cases()] + experiment_cases()
        for scenario in scenarios:
            for key, value in scenario.dictionary().items():
                self.assertTrue(value is None or np.isscalar(value), f'{key} = {value!r}')

    def test_pause_takes_real_time(self):
        world, _ = self.fatigue_path(Scenario(), 0, steps=1, episodes=5)
        self.assertTrue(np.allclose(world.results()['seconds'], 10))
        legacy = fast_stress_test(Scenario()).pause_duration(self.calibration['public_mu'])
        self.assertAlmostEqual(legacy, .1 * np.exp(float(self.calibration['public_mu'])))

    def test_falcon_fatigue_starts_fresh_and_stays_in_range(self):
        _, path = self.fatigue_path(Scenario(model='FALCON', beta=1), 3)
        self.assertLess(np.abs(path[0]).max(), 1e-9)  # warm-up is not fatigue
        self.assertGreaterEqual(path.min(), 0)
        self.assertLessEqual(path.max(), 4)
        self.assertTrue((np.diff(path, axis=0) >= -1e-9).all())  # no recovery without falcon_recovery

    def test_falcon_planner_matches_the_population_on_average(self):
        _, path = self.fatigue_path(Scenario(model='FALCON', beta=1), 3, episodes=1000)
        table = transition_table(self.contexts['difficulty'][:1], Scenario(model='FALCON', beta=1))
        belief = np.eye(5)[0]
        for _ in range(150):
            belief = belief @ table[0, :, 3, :]
        self.assertLess(abs(belief @ np.arange(5) - path[-1].mean()), .5)

    def test_forced_lapse_reaches_every_session(self):
        world, _ = self.fatigue_path(Scenario(model='Lapse', beta=1), 1)
        self.assertTrue(world.lapsed().all())

    def test_deployable_planner_does_not_copy_the_true_world(self):
        world = Scenario(model='FALCON', beta=3, recovery=.5, signal=2., fatigue_scale=1., lam=.4)
        assumed = deployable_assumption(world)
        self.assertEqual((assumed.model, assumed.beta, assumed.recovery, assumed.signal, assumed.fatigue_scale),
                         ('Markov', 1., .35, 1., .1))
        self.assertEqual((assumed.lam, assumed.steps), (world.lam, world.steps))  # design choices are known

    def test_speed_tracker_learns_personal_speed_then_stops(self):
        tracker = SpeedTracker(2, prior_sd=.2, noise_sd=.4, answers=20)
        for _ in range(30):
            tracker.update(np.array([.3, -.3]), np.zeros(2), np.array([True, True]))
        expected = .3 * 20 / (20 + 4)
        self.assertTrue(np.allclose(tracker.offset, [expected, -expected]))
        before = tracker.offset.copy()
        tracker.update(np.array([5., 5.]), np.zeros(2), np.array([True, True]))
        self.assertTrue(np.allclose(tracker.offset, before))
        tracker = SpeedTracker(1, prior_sd=.2, noise_sd=.4)
        tracker.update(np.array([5.]), np.zeros(1), np.array([False]))  # pauses never count
        self.assertEqual(tracker.offset[0], 0)

    def test_rest_threshold_pauses_only_when_tired(self):
        model = PlanningModel(self.contexts, Scenario(beta=2), population_model(self.calibration), plan=False)
        model.rest_threshold = 2.
        indices = np.zeros(2, dtype=int)
        beliefs = np.array([[1., 0, 0, 0, 0], [0, 0, 0, 0, 1.]])
        actions = model.action(indices, beliefs, 0, 'rest_threshold')
        self.assertNotEqual(actions[0], 0)
        self.assertEqual(actions[1], 0)

    def test_time_to_target_ignores_unreached_runs(self):
        rows = []
        for seed, (qmdp_time, other_time) in enumerate([(1000, 1500), (2400, 1200), (900, 2400)]):
            for policy, time in [('qmdp', qmdp_time), ('verify', other_time)]:
                rows.append({'model': 'Markov', 'beta': 1, 'seed_size': 200, 'seed': seed, 'policy': policy,
                    'reached_80': time < 2400, 'time_80_capped': time,
                    **{f'accuracy_{b}s': 70. for b in [300, 600, 1200, 2400]}})
        _, times = comparison_tables(pd.DataFrame(rows))
        self.assertEqual(times.pairs_both_reached.iloc[0], 1)
        self.assertEqual(times.seconds_saved_mean.iloc[0], 500)


if __name__ == '__main__':
    unittest.main()
