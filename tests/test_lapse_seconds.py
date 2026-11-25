"""Checks for the seconds-onset Lapse mode added on 21 Nov 2026 ('forced_seconds').

The answer-count mode ('forced') must behave exactly as before; the new mode must start the
lapse at its scheduled second, share its onset draw across policies, and count pauses as time.
"""
import importlib.util
import unittest
import numpy as np
from src.common import load_calibration, load_contexts
from src.sim.fatigue import Scenario
from src.sim.world import AnnotationWorld
import src.eval.stage1 as stage1
import src.eval.study as study

HAS_TORCH = importlib.util.find_spec('torch') is not None

# Recorded from the code before the change (21 Nov 2026): Stage-1 run_policy, validation people,
# 40 sessions, seed 11. (beta, policy): (mean reward, sessions lapsed, action counts).
BEFORE_CHANGE = {
    (1, 'verify'): (-82.275981, 40, [0, 6000, 0, 0, 0]),
    (1, 'qmdp'): (-114.201083, 40, [158, 4921, 94, 827, 0]),
    (1, 'blind'): (-396.03607, 40, [0, 1012, 2215, 2773, 0]),
    (1, 'full'): (-863.486263, 40, [0, 0, 0, 6000, 0]),
    (0, 'verify'): (-58.211027, 40, [0, 6000, 0, 0, 0]),
    (0, 'qmdp'): (-137.146896, 40, [0, 1012, 2215, 2773, 0]),
    (0, 'full'): (-211.603429, 40, [0, 0, 0, 6000, 0])}


class LapseSecondsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_calibration()
        cls.contexts = load_contexts()

    def world(self, scenario, episodes=10, seed=42):
        return AnnotationWorld(self.calibration, self.contexts, scenario, episodes=episodes, seed=seed)

    def test_answer_count_mode_is_unchanged(self):
        for (beta, policy), (reward, lapsed, counts) in BEFORE_CHANGE.items():
            result = stage1.run_policy(self.contexts, Scenario(model='Lapse', beta=beta), policy,
                                       'validation', 40, 11, calibration=self.calibration)
            self.assertAlmostEqual(float(result['reward'].mean()), reward, places=4, msg=(beta, policy))
            self.assertEqual(int(result['lapsed'].sum()), lapsed)
            self.assertEqual(result['action_counts'].sum(0).tolist(), counts, msg=(beta, policy))

    def test_seconds_mode_starts_at_its_scheduled_second(self):
        world = self.world(Scenario(model='Lapse', beta=0, lapse_mode='forced_seconds',
                                    lapse_onset_min=5., lapse_onset_max=5.))
        for _ in range(40):
            world.step(np.ones(10, dtype=int))
            # Lapsed exactly when the session's clock has reached 5 s. The answer that crosses
            # 5 s was given while still engaged; the lapse applies from the next question.
            np.testing.assert_array_equal(world.lapsed(), world.seconds >= 5.)
        self.assertTrue(world.lapsed().all())

    def test_onset_draw_is_shared_across_policies(self):
        scenario = Scenario(model='Lapse', beta=1, lapse_mode='forced_seconds', lapse_onset_min=20., lapse_onset_max=60.)
        verify, full = self.world(scenario, episodes=50), self.world(scenario, episodes=50)
        np.testing.assert_array_equal(verify._lapse_onset, full._lapse_onset)
        answered = {}
        for name, world, action in [('verify', verify, 1), ('full', full, 3)]:
            count = np.zeros(50)
            for _ in range(150):
                count += ~world.lapsed()
                world.step(np.full(50, action))
            answered[name] = count
        self.assertTrue(verify.lapsed().all() and full.lapsed().all())
        # Same onset in seconds: slower full labels reach it after fewer answers.
        self.assertLess(answered['full'].mean(), answered['verify'].mean())

    def test_pauses_count_as_time_only_in_seconds_mode(self):
        seconds_world = self.world(Scenario(model='Lapse', beta=1, lapse_mode='forced_seconds',
                                            lapse_onset_min=25., lapse_onset_max=25.))
        count_world = self.world(Scenario(model='Lapse', beta=1, lapse_onset_min=3., lapse_onset_max=3.))
        for pauses in range(1, 6):
            seconds_world.step(np.zeros(10, dtype=int))
            count_world.step(np.zeros(10, dtype=int))
            self.assertEqual(bool(seconds_world.lapsed().all()), pauses * 10 >= 25)
            self.assertFalse(count_world.lapsed().any())   # pauses are not answers

    def test_stage2_onset_range_reaches_every_shift(self):
        world = self.world(Scenario(model='Lapse', beta=1, lapse_mode='forced_seconds',
                                    lapse_onset_min=75., lapse_onset_max=310.), episodes=20)
        self.assertTrue(((world._lapse_onset >= 75.) & (world._lapse_onset <= 310.)).all())
        for _ in range(32):                                   # 32 pauses = 320 s < one 600-s shift
            world.step(np.zeros(20, dtype=int))
        self.assertTrue(world.lapsed().all())

    def test_stage1_case_lists(self):
        initial = study.initial_cases()
        self.assertEqual(len(initial), 28)                     # unchanged
        self.assertTrue(all(scenario.lapse_mode == 'forced' for _, scenario, _ in initial if scenario.model == 'Lapse'))
        extra = study.lapse_seconds_cases()
        self.assertEqual([name for name, _, _ in study.final_test_cases()], [name for name, _, _ in initial + extra])
        for _, scenario, _ in extra:
            self.assertEqual((scenario.lapse_mode, scenario.lapse_onset_min, scenario.lapse_onset_max),
                             ('forced_seconds', 14., 88.))

    @unittest.skipUnless(HAS_TORCH, 'Stage-2 module needs PyTorch')
    def test_stage2_case_lists(self):
        import src.eval.stage2 as stage2
        development = stage2.experiment_cases()
        self.assertEqual(len(development), 5)                  # unchanged
        lapse = development[-1]
        self.assertEqual((lapse.lapse_mode, lapse.lapse_onset_min, lapse.lapse_onset_max), ('forced', 100., 300.))
        final = stage2.final_test_cases()
        self.assertEqual(final[:5], development)
        self.assertEqual((final[5].model, final[5].beta, final[5].lapse_mode, final[5].lapse_onset_min,
                          final[5].lapse_onset_max), ('Lapse', 1., 'forced_seconds', 75., 310.))


if __name__ == '__main__':
    unittest.main()
