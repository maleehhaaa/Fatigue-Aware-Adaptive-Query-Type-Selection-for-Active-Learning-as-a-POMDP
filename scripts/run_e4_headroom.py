"""E4: oracle headroom. How much better than greedy could a controller do, and how much of
that does QMDP (and each other controller) capture? Analysis only: no new runs.

For each Stage-1 case of a study folder (all_validation_episodes.csv.gz):
  headroom        = oracle - greedy                 (oracle: true fatigue + the same planner)
  gap(X)          = X - greedy
  share(X)        = gap(X) / headroom               for QMDP, greedy+rest, grid, DQN, deployable QMDP
  ceiling         = oracle - best fixed strategy    (best of the 8 required fixed/simple baselines)
and, across Markov beta = 0.5, 1, 2, 3 (H3 restated), the least-squares slope against beta of
QMDP's gap, the headroom and QMDP's share. Intervals: 95% bootstrap over the 500 sessions
(4,000 resamples, seed 42). Sessions are resampled jointly across policies and cases, because
every case uses the same session seed.

The oracle is a reference, not an upper bound: it plans with the true fatigue level rounded to
the planner's 5 states, and it can be beaten (in FALCON, always-verify beats it at beta >= 2).

    python scripts/run_e4_headroom.py              # the three lambda versions below
    python scripts/run_e4_headroom.py --study outputs/study/<id>/seed-42 --label my_label

Results go to outputs/e4_headroom/.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

project = Path(__file__).resolve().parents[1]
OUT = project / 'outputs/e4_headroom'
STUDIES = {  # label: (folder, note)
    'lambda_0.25': (project / 'outputs/study/68ff499a2c/seed-42', 'Mac, 20 Nov rerun (pre-set lambda)'),
    # Copied into the project on 22 Nov from the teammate's separate aml_lambda075 folder
    # (byte-identical; see outputs/imported/aml_lambda075/PROVENANCE.json).
    'lambda_0.75': (project / 'outputs/imported/aml_lambda075/study/b461149a97/seed-42',
                    'Windows (teammate), 20 Nov; a different machine, compare patterns only'),
    'lambda_1.0': (project / 'outputs/study/3585ddc504/seed-42', 'Mac, 21 Nov rerun (chosen lambda)')}
FIXED = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload', 'blind']
CONTROLLERS = ['qmdp', 'rest_threshold', 'grid', 'dqn', 'qmdp_fixed']
MARKOV = ['Markov_beta_0.5', 'Markov_beta_1', 'Markov_beta_2', 'Markov_beta_3']
BETAS = np.array([.5, 1., 2., 3.])
RESAMPLES, SEED = 4000, 42


def load(folder):
    table = pd.read_csv(folder / 'all_validation_episodes.csv.gz', keep_default_na=False, na_values=[''],
                        usecols=['case', 'policy', 'episode', 'reward', 'model', 'beta'])
    return {case: part.pivot(index='episode', columns='policy', values='reward').sort_index()
            for case, part in table.groupby('case')}


def case_measures(wide):
    """Point values for one case (wide: sessions x policies)."""
    mean = wide.mean()
    fixed = [p for p in FIXED if p in mean]
    best_fixed = mean[fixed].idxmax()
    out = {'greedy': mean['myopic'], 'oracle': mean['oracle'], 'headroom': mean['oracle'] - mean['myopic'],
           'best_fixed': best_fixed, 'best_fixed_reward': mean[best_fixed],
           'ceiling_oracle_minus_best_fixed': mean['oracle'] - mean[best_fixed],
           'qmdp_minus_best_fixed': mean['qmdp'] - mean[best_fixed]}
    for name in CONTROLLERS:
        if name in mean:
            out[f'gap_{name}'] = mean[name] - mean['myopic']
            out[f'share_{name}'] = (mean[name] - mean['myopic']) / out['headroom'] if out['headroom'] > 1e-9 else np.nan
    return out


def bootstrap_cases(cases, indices):
    """Resampled means for every case, using the same session indices everywhere."""
    return {case: wide.to_numpy()[indices].mean(1) for case, wide in cases.items()}, \
           {case: list(wide.columns) for case, wide in cases.items()}


def analyse(label, folder, note):
    cases = load(folder)
    sessions = len(next(iter(cases.values())))
    random = np.random.default_rng(SEED)
    draws = random.integers(sessions, size=(RESAMPLES, sessions))
    rows = []
    for case, wide in cases.items():
        point = case_measures(wide)
        values = wide.to_numpy()[draws].mean(1)              # resamples x policies
        column = {p: i for i, p in enumerate(wide.columns)}
        greedy, oracle = values[:, column['myopic']], values[:, column['oracle']]
        headroom = oracle - greedy
        intervals = {'headroom': headroom}
        for name in CONTROLLERS:
            if name in column:
                gap = values[:, column[name]] - greedy
                intervals[f'gap_{name}'] = gap
                intervals[f'share_{name}'] = np.where(np.abs(headroom) > 1e-9, gap / headroom, np.nan)
        fixed_index = column[point['best_fixed']]
        intervals['ceiling_oracle_minus_best_fixed'] = oracle - values[:, fixed_index]
        intervals['qmdp_minus_best_fixed'] = values[:, column['qmdp']] - values[:, fixed_index]
        row = {'study': label, 'case': case, **point}
        for key, sample in intervals.items():
            low, high = np.nanquantile(sample, [.025, .975]) if np.isfinite(sample).any() else (np.nan, np.nan)
            row[f'{key}_ci_low'], row[f'{key}_ci_high'] = low, high
        rows.append(row)
    table = pd.DataFrame(rows)
    # H3 restated: slopes against beta over the four core Markov cases, sessions resampled jointly.
    trends = []
    if all(case in cases for case in MARKOV):
        def slopes(sample_means):
            out = {}
            for key in ['gap_qmdp', 'headroom', 'share_qmdp', 'gap_rest_threshold', 'share_rest_threshold']:
                y = np.array([sample_means[case][key] for case in MARKOV])
                out[key] = np.polyfit(BETAS, y, 1)[0]
            return out
        point = slopes({case: case_measures(cases[case]) for case in MARKOV})
        boot = {key: [] for key in point}
        for indices in draws:
            means = {}
            for case in MARKOV:
                wide = cases[case]
                means[case] = case_measures(pd.DataFrame(wide.to_numpy()[indices], columns=wide.columns))
            for key, value in slopes(means).items():
                boot[key].append(value)
        for key, value in point.items():
            low, high = np.nanquantile(boot[key], [.025, .975])
            trends.append({'study': label, 'measure': key, 'slope_per_beta': value, 'ci_low': low, 'ci_high': high,
                           'values_beta_0.5_1_2_3': [round(case_measures(cases[c])[key], 3) for c in MARKOV]})
    shown = folder.relative_to(project).as_posix() if folder.is_relative_to(project) else str(folder)
    return table, pd.DataFrame(trends), {'label': label, 'folder': shown, 'note': note, 'sessions': sessions}


def main(studies):
    OUT.mkdir(parents=True, exist_ok=True)
    tables, trends, notes = [], [], []
    for label, (folder, note) in studies.items():
        folder = Path(folder)
        if not (folder / 'all_validation_episodes.csv.gz').exists():
            print(f'Skipping {label}: {folder} not found')
            continue
        table, trend, info = analyse(label, folder, note)
        tables.append(table), trends.append(trend), notes.append(info)
        print(f'{label}: {len(table)} cases', flush=True)
    table, trend = pd.concat(tables, ignore_index=True), pd.concat(trends, ignore_index=True)
    table.round(4).to_csv(OUT / 'headroom_by_case.csv', index=False)
    trend.round(4).to_csv(OUT / 'h3_trends_vs_beta.csv', index=False)
    (OUT / 'sources.json').write_text(json.dumps(notes, indent=2))
    show = table[table.case.isin(MARKOV + ['FALCON_beta_1', 'FALCON_beta_2', 'FALCON_beta_3', 'Lapse_beta_1', 'no_fatigue'])]
    print(show[['study', 'case', 'headroom', 'gap_qmdp', 'share_qmdp', 'share_rest_threshold', 'best_fixed',
                'ceiling_oracle_minus_best_fixed', 'qmdp_minus_best_fixed']].round(3).to_string(index=False))
    print(trend.round(3).to_string(index=False))
    plot(table)


def plot(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for label, part in table[table.case.isin(MARKOV)].groupby('study'):
        part = part.set_index('case').loc[MARKOV]
        axes[0].plot(BETAS, part.headroom, marker='o', label=label)
        axes[1].plot(BETAS, part.gap_qmdp, marker='o', label=label)
        axes[2].errorbar(BETAS, part.share_qmdp, yerr=[part.share_qmdp - part.share_qmdp_ci_low,
                         part.share_qmdp_ci_high - part.share_qmdp], marker='o', capsize=2, label=label)
    axes[0].set(title='Headroom: oracle - greedy', xlabel='beta', ylabel='net reward')
    axes[1].set(title='QMDP - greedy', xlabel='beta')
    axes[2].set(title="QMDP's share of the headroom", xlabel='beta', ylim=(0, 1))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'headroom_markov.png', dpi=150)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', help='one study folder (…/seed-42); default: the three lambda versions')
    parser.add_argument('--label', default='custom')
    arguments = parser.parse_args()
    main({arguments.label: (project / arguments.study, 'given on the command line')} if arguments.study else STUDIES)
