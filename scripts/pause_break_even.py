"""When does a pause pay? Break-even test for QMDP's pause decision.

QMDP pauses only if the value of a pause beats the value of the best question about the
current image. In QMDP's own model (src/control/qmdp.py):

    Q(pause, F)   = -lam * pause_seconds + r * V[t+1](F-1) + (1 - r) * V[t+1](F)
    Q(ask a, F, x) = I(x, F, a) - lam * seconds(x, F, a) + sum_F' P(F' | F, a, x) * V[t+1](F')

Two parts:

  theory    Rebuilds QMDP's value tables (the same PlanningModel the controller uses) for each
            lambda, pause length and recovery rate, and reports, per fatigue level F:
              - the value of one fatigue level, V(F-1) - V(F), in bits
              - the expected gain of one pause, r * (V(F-1) - V(F)), against its cost, lam * seconds
              - the share of images for which pausing beats every question, if F were known
              - the pause advantage, Q(pause) - max_a Q(ask a), averaged over images
              - the smallest belief P(F = k) (rest of the belief on F = 0) at which QMDP pauses
            Also the break-even lambda and pause length for each fatigue level.
  observed  Reads existing Stage-2 answer records (no new runs) and reports how often QMDP and
            greedy-plus-rest actually paused, by estimated fatigue, for each lambda in the sweep
            and in the development study.

    python scripts/pause_break_even.py theory
    python scripts/pause_break_even.py observed
    python scripts/pause_break_even.py smoke      # small grid, separate folder; not a result

Uses the Stage-1 contexts (notebook 04) and training-population facts only. src/ is not
changed. Results go to outputs/pause_break_even/.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
from src.common import load_calibration, load_contexts
from src.control.qmdp import PlanningModel
from src.eval.stage1 import population_model
from src.sim.fatigue import Scenario

OUT = project / 'outputs/pause_break_even'

# ---- Settings (change here) ----
WORLDS = {'Markov0.5': Scenario(beta=.5), 'Markov1': Scenario(beta=1), 'Markov2': Scenario(beta=2),
          'Markov3': Scenario(beta=3),
          # Control: the planner's FALCON model has no recovery, so a pause should never pay.
          'FALCON1': Scenario(model='FALCON', beta=1)}
LAMBDAS = (.25, .5, .75, .875, 1., 1.25)
PAUSES = (5., 10., 20.)
RECOVERIES = (.2, .35, .5)
SHIFT_SECONDS = 600.        # Stage 2 plans over remaining_time / pause_seconds steps (al_loop.py)
STAGE1_STEPS = 150
P_GRID = np.linspace(0, 1, 101)


def horizons(pause_seconds):
    """Stage-1 horizon, and the Stage-2 horizon at the start of a 600-s shift (capped at 150)."""
    return {'stage1': STAGE1_STEPS, 'stage2_shift_start': int(min(STAGE1_STEPS, SHIFT_SECONDS / pause_seconds))}


def pause_analysis(contexts, population, scenario, steps):
    """Pause vs best question at the first step of a horizon of `steps`, from QMDP's own tables."""
    world = replace(scenario, steps=steps)
    model = PlanningModel(contexts, world, population)
    future = np.einsum('cfag,g->cfa', model.transition, model.value[1])
    q = model.reward + future                                  # contexts x fatigue x action
    q_pause, q_ask = q[:, :, 0], q[:, :, 1:4]                  # gold (4) is never used by QMDP
    advantage = q_pause - q_ask.max(2)                         # > 0: pause beats every question
    rows = []
    for fatigue in range(5):
        level_value = model.value[1][fatigue - 1] - model.value[1][fatigue] if fatigue else 0.
        # Smallest P(F = fatigue), rest on F = 0, at which pause is QMDP's choice (per image).
        mixed = ((1 - P_GRID)[:, None, None] * q[None, :, 0, :4] + P_GRID[:, None, None] * q[None, :, fatigue, :4])
        pauses = mixed.argmax(2) == 0                          # p x contexts
        first = np.where(pauses.any(0), pauses.argmax(0), -1)
        threshold = np.where(first >= 0, P_GRID[np.clip(first, 0, None)], np.nan)
        rows.append({'fatigue': fatigue, 'value_of_one_level_bits': level_value,
                     'pause_gain_bits': world.recovery * level_value if world.model != 'FALCON' else 0.,
                     'pause_cost_bits': world.lam * world.pause_duration(population['mu']),
                     'share_images_pause_optimal': float((advantage[:, fatigue] > 0).mean()),
                     'mean_pause_advantage_bits': float(advantage[:, fatigue].mean()),
                     'median_belief_threshold': float(np.nanmedian(threshold)) if np.isfinite(threshold).any() else np.nan,
                     'share_images_pause_reachable': float(np.isfinite(threshold).mean())})
    return rows


def break_even(table, key, by):
    """Linear interpolation of where the mean pause advantage crosses zero along `key`."""
    out = []
    for group, part in table.groupby(by):
        part = part.sort_values(key)
        x, y = part[key].to_numpy(), part.mean_pause_advantage_bits.to_numpy()
        crossing = np.nan
        for i in range(len(x) - 1):
            if np.sign(y[i]) != np.sign(y[i + 1]) and y[i] != y[i + 1]:
                crossing = x[i] + (x[i + 1] - x[i]) * (0 - y[i]) / (y[i + 1] - y[i])
                break
        if np.isnan(crossing):
            crossing = (f'pause pays at every {key} tested' if (y > 0).all()
                        else f'pause never pays at any {key} tested' if (y <= 0).all() else np.nan)
        out.append({**dict(zip(by, group)), f'break_even_{key}': crossing})
    return pd.DataFrame(out)


def run_theory(worlds=WORLDS, lambdas=LAMBDAS, pauses=PAUSES, recoveries=RECOVERIES):
    OUT.mkdir(parents=True, exist_ok=True)
    contexts, calibration = load_contexts(), load_calibration()
    population = population_model(calibration)
    rows = []
    for name, base in worlds.items():
        for lam in lambdas:
            for pause in pauses:
                for recovery in recoveries:
                    scenario = replace(base, lam=float(lam), pause_seconds=float(pause), recovery=float(recovery))
                    for horizon_name, steps in horizons(pause).items():
                        for row in pause_analysis(contexts, population, scenario, steps):
                            rows.append({'world': name, 'lam': lam, 'pause_seconds': pause, 'recovery': recovery,
                                         'horizon': horizon_name, 'steps': steps, **row})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'pause_value_by_fatigue.csv', index=False)
    tired = table[table.fatigue >= 1]
    by = ['world', 'pause_seconds', 'recovery', 'horizon', 'fatigue']
    break_even(tired, 'lam', by).to_csv(OUT / 'break_even_lambda.csv', index=False)
    break_even(tired, 'pause_seconds', ['world', 'lam', 'recovery', 'horizon', 'fatigue']).to_csv(
        OUT / 'break_even_pause_seconds.csv', index=False)
    # Headline: the study's settings (pause 10 s, recovery 0.35), Stage-2 horizon.
    head = table[(table.pause_seconds == 10) & np.isclose(table.recovery, .35) & (table.horizon == 'stage2_shift_start')]
    head = head.pivot_table(index=['world', 'fatigue'], columns='lam',
                            values=['mean_pause_advantage_bits', 'share_images_pause_optimal']).round(3)
    head.to_csv(OUT / 'headline_pause10_r035_stage2.csv')
    (OUT / 'settings.json').write_text(json.dumps({'worlds': list(worlds), 'lambdas': list(lambdas),
        'pauses': list(pauses), 'recoveries': list(recoveries), 'contexts': 'Stage-1 contexts (notebook 04)',
        'population': 'training people only', 'horizons': 'stage1 = 150 steps; stage2 = 600 s / pause seconds, capped at 150'}, indent=2))
    print(head.to_string())


# ------------------------------------------------------------------ observed (existing runs only)
def read_audit(folder):
    folder = Path(str(folder).replace('\\', '/'))
    path = folder if folder.is_absolute() else project / folder
    if 'advanced_ml/' in str(folder):
        path = project / str(folder).split('advanced_ml/')[-1]
    return pd.read_csv(path / 'answer_audit.csv.gz', keep_default_na=False, na_values=[''],
                       usecols=['action', 'estimated_fatigue', 'fatigue'])


def pause_rates(runs, label):
    rows = []
    bins = [-.01, .5, 1.5, 2.5, 5]
    names = ['est F < 0.5', '0.5-1.5', '1.5-2.5', '> 2.5']
    for keys, part in runs.groupby(['world', 'policy', 'lam']):
        audits = pd.concat([read_audit(f) for f in part.run_folder], ignore_index=True)
        audits['bin'] = pd.cut(audits.estimated_fatigue, bins, labels=names)
        rate = audits.groupby('bin', observed=False).action.apply(lambda a: (a == 0).mean())
        share = audits.bin.value_counts(normalize=True).reindex(names)
        rows.append({'source': label, 'world': keys[0], 'policy': keys[1], 'lam': keys[2],
                     'pause_rate_overall': (audits.action == 0).mean(),
                     'true_F_when_paused': audits.loc[audits.action == 0, 'fatigue'].mean(),
                     **{f'pause_rate {n}': rate.get(n) for n in names},
                     **{f'share of turns {n}': share.get(n) for n in names}})
    return pd.DataFrame(rows)


def run_observed():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = []
    sweep = project / 'outputs/lambda_sweep/runs.csv'
    if sweep.exists():
        runs = pd.read_csv(sweep, keep_default_na=False, na_values=[''])
        runs['lam'] = pd.to_numeric(runs.lam, errors='coerce')
        runs = runs[runs.policy.isin(['qmdp', 'rest_threshold']) & runs.world.isin(['Markov1', 'Markov2', 'Lapse1'])]
        tables.append(pause_rates(runs, 'sweep (training people, seeds 100-109)'))
    pointer = project / 'outputs/stage2_latest.json'
    if pointer.exists():
        folder = project / json.loads(pointer.read_text())['folder'].replace('\\', '/')
        study = pd.read_csv(folder / 'all_runs.csv', keep_default_na=False, na_values=[''])
        lam = json.loads((folder / 'config.json').read_text())['scenarios'][0]['lam']
        study = study[study.policy.isin(['qmdp', 'rest_threshold']) & study.model.isin(['Markov', 'Lapse'])]
        study = study.assign(world=study.model + study.beta.astype(str), lam=lam)
        tables.append(pause_rates(study, f'development study ({folder.parent.name})'))
    table = pd.concat(tables).round(4)
    table.to_csv(OUT / 'observed_pause_rates.csv', index=False)
    print(table.to_string(index=False))


def smoke():
    """Tiny version, separate folder. Not a result."""
    global OUT
    OUT = project / 'outputs/pause_break_even_smoke'
    run_theory(worlds={'Markov1': Scenario(beta=1), 'FALCON1': Scenario(model='FALCON', beta=1)},
               lambdas=(.75, 1.), pauses=(10.,), recoveries=(.35,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['theory', 'observed', 'smoke'])
    {'theory': run_theory, 'observed': run_observed, 'smoke': smoke}[parser.parse_args().part]()
