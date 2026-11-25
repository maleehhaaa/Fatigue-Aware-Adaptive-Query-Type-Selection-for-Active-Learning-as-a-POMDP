"""Give Stage 2 a public answer function; keep truth and audit data separate."""
import numpy as np
from src.sim.world import AnnotationWorld


def make_session_factory(calibration, scenario, seed, split='validation', max_steps=10000):
    audit = []
    def new_session(session_number):
        # Fixed dummy context prevents pool-size differences from shifting random tapes.
        dummy = {'image_id': np.array([0]), 'probabilities': np.ones((1, 10)) / 10,
                 'difficulty': np.zeros(1)}
        world = AnnotationWorld(calibration, dummy, scenario, split=split, episodes=1,
            seed=seed * 10000 + session_number + 700000, max_steps=max_steps, practice_cap=200)
        def ask(contexts, index, action):
            world.contexts = contexts
            world.indices[world.step_number, 0] = index
            lapsed = bool(world.lapsed()[0])  # state when the answer is given (evaluation only)
            observation = world.step(np.array([action]))
            audit.append({'call': len(audit), 'session': session_number,
                'person_id': int(calibration['person_ids'][world._person[0]]),
                'correct': bool(world.last['correct'][0]), 'fatigue': float(world.last['fatigue'][0]),
                'lapsed': lapsed})
            return observation
        return ask
    return new_session, audit
