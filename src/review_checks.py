"""Checks for the 19 Nov 2026 review fixes. Run from the project root:

    .venv/bin/python -m src.review_checks

Each check can fail for the reason it names. Uses only the saved simulator calibration.
"""
import numpy as np
from src.common import load_calibration
from src.sim.fatigue import Scenario, fast_stress_test, transition_table
from src.sim.world import AnnotationWorld


def dummy_contexts(calibration, count=500, seed=0):
    random = np.random.default_rng(seed)
    ids = np.flatnonzero(calibration['difficulty'] != 0)[:count]
    logits = random.normal(0, 1.2, (len(ids), 10))
    logits[np.arange(len(ids)), calibration['true_labels'][ids]] += 1.5
    probabilities = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return {'image_id': ids, 'probabilities': probabilities,
            'difficulty': np.full(len(ids), calibration['difficulty'][ids].mean())}


def run(calibration, contexts, scenario, action, steps=150, episodes=1000, seed=1):
    world = AnnotationWorld(calibration, contexts, scenario, 'validation', episodes, seed)
    fatigue = []
    for _ in range(steps):
        fatigue.append(world.oracle_fatigue())
        world.step(np.full(episodes, action))
    return world, np.array(fatigue)


def main():
    calibration = load_calibration()
    contexts = dummy_contexts(calibration)
    checks = []

    _, fatigue = run(calibration, contexts, Scenario(beta=2), 3)
    checks.append(('Realistic: always-full people are not exhausted within 50 questions',
                   fatigue[49].mean() < 2, f'mean F after 50 = {fatigue[49].mean():.2f}'))
    _, fast = run(calibration, contexts, fast_stress_test(Scenario(beta=2)), 3)
    checks.append(('Stress preset reproduces the old ~15-question exhaustion',
                   fast[20].mean() > 3, f'mean F after 20 = {fast[20].mean():.2f}'))

    world, _ = run(calibration, contexts, Scenario(beta=2), 0, steps=1)
    pause = world.results()['seconds'].mean()
    checks.append(('A pause takes real time (10 s by default)', abs(pause - 10) < 1e-9, f'{pause:.2f} s'))

    _, falcon = run(calibration, contexts, Scenario(model='FALCON', beta=1), 3)
    checks.append(('FALCON fatigue starts at 0 (warm-up is not fatigue)',
                   np.abs(falcon[0]).max() < 1e-9, f'max F at start = {np.abs(falcon[0]).max():.1e}'))
    checks.append(('FALCON fatigue stays within 0-4', falcon.min() >= 0 and falcon.max() <= 4,
                   f'range {falcon.min():.2f} to {falcon.max():.2f}'))
    table = transition_table(contexts['difficulty'][:1], Scenario(model='FALCON', beta=1))
    belief = np.eye(5)[0]
    for _ in range(150):
        belief = belief @ table[0, :, 3, :]
    planned, actual = belief @ np.arange(5), falcon[-1].mean()
    checks.append(('FALCON planner expects about as much fatigue as people get',
                   abs(planned - actual) < .5, f'planner {planned:.2f} vs world {actual:.2f}'))

    world, _ = run(calibration, contexts, Scenario(model='Lapse', beta=1), 1)
    checks.append(('Forced lapse happens in every session', world.lapsed().all(),
                   f'{world.lapsed().mean():.0%} of sessions'))

    for model in ['None', 'Markov', 'FALCON', 'Lapse']:
        rows = transition_table(contexts['difficulty'][:20], Scenario(model=model, beta=1)).sum(-1)
        checks.append((f'{model} planning rows are probabilities', np.allclose(rows, 1), ''))
    no_recovery = transition_table(contexts['difficulty'][:20], Scenario(model='FALCON', beta=1))[:, :, 0]
    checks.append(('FALCON pauses do not recover by default', np.allclose(no_recovery, np.eye(5)), ''))

    failed = 0
    for name, passed, detail in checks:
        failed += not passed
        print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")
    print(f'\n{len(checks) - failed}/{len(checks)} review checks passed.')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
