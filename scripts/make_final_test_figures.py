"""Figures and tables for the report, made from the final-test outputs only.

Reads outputs/final_test (written once by scripts/run_final_test.py) and, for the
development-vs-test comparison, outputs/study/cbf5a08c06 and outputs/stage2_study/1e2d6d55c0.
It runs no experiments and does not touch the test data again: it only reads saved CSVs and the
saved accuracy curves of the 2,400 final-test Stage-2 runs.

Usage (from the project folder):
    python scripts/make_final_test_figures.py
Output: outputs/final_test/figures/*.png and *.pdf, plus outputs/final_test/figures/tables.md
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'outputs' / 'final_test'
OUT = FINAL / 'figures'
DEV_STAGE1 = ROOT / 'outputs' / 'study' / 'cbf5a08c06' / 'seed-42' / 'condition_summary.csv'
DEV_STAGE2 = ROOT / 'outputs' / 'stage2_study' / '1e2d6d55c0' / 'seed-42' / 'all_runs.csv'

# ---------- names (same wording as notebook 10) and one fixed colour per policy ----------
NAMES = {'qmdp': 'QMDP', 'qmdp_fixed': 'QMDP (fixed model)', 'myopic': 'Greedy', 'rest_threshold': 'Greedy + rest',
         'blind': 'Ignores fatigue', 'verify': 'Always verify', 'two_choice': 'Always 2-choice', 'full': 'Always full',
         'random': 'Random type', 'random_image_full': 'Random image + full', 'alternating': 'Alternating',
         'exp_decay': 'Exp. decay', 'gap_workload': 'Gap workload', 'grid': 'Grid planner', 'dqn': 'DQN', 'oracle': 'Oracle'}
# Validated categorical order (dataviz reference palette, light mode): adjacent CVD dE >= 9.1.
# Three slots sit below 3:1 contrast, so every chart also uses marker shape / line style and a legend.
COLOR = {'qmdp': '#2a78d6', 'rest_threshold': '#eb6834', 'myopic': '#1baf7a', 'blind': '#eda100',
         'two_choice': '#e87ba4', 'verify': '#6250d6', 'dqn': '#008300'}
MARKER = {'qmdp': 'o', 'rest_threshold': 's', 'myopic': '^', 'blind': 'D', 'two_choice': 'v', 'verify': 'P', 'dqn': 'X'}
DASH = {'qmdp': '-', 'rest_threshold': '--', 'myopic': '-.', 'blind': ':', 'two_choice': (0, (5, 1, 1, 1)),
        'verify': (0, (3, 1)), 'dqn': (0, (1, 1))}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
AHEAD, BEHIND, UNCLEAR = '#b7d3f6', '#f6c4c3', '#f0efec'
SETTINGS = ['None0', 'Markov1', 'Markov2', 'FALCON1', 'Lapse1:forced', 'Lapse1:forced_seconds']
SETTING_LABEL = {'None0': 'No fatigue', 'Markov1': 'Markov, β = 1', 'Markov2': 'Markov, β = 2',
                 'FALCON1': 'FALCON, β = 1', 'Lapse1:forced': 'Lapse, β = 1\n(onset in answers)',
                 'Lapse1:forced_seconds': 'Lapse, β = 1\n(onset in seconds)'}
CURVE_POLICIES = ['qmdp', 'rest_threshold', 'myopic', 'blind', 'two_choice', 'verify']

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
    'axes.axisbelow': True, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'lines.linewidth': 2})


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def status(row, p='p_holm'):
    if row[p] < .05:
        return 'ahead' if row.difference > 0 else 'behind'
    return 'unclear'


def check_inputs():
    for part in ['stage1', 'stage2', 'grid']:
        assert (FINAL / f'{part}_complete.json').exists(), f'final test part {part} is not complete'
    report = json.loads((FINAL / 'final_report.json').read_text())
    assert report['stage2']['final_test_evaluated'] is True
    return report


# ---------- Figure 1: the 12 primary Stage-2 comparisons ----------
def fig_primary():
    p = read(FINAL / 'stage2_primary.csv')
    order = [(s, n) for s in SETTINGS for n in (50, 200)]
    p = p.set_index(['setting', 'seed_size']).loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    y = np.arange(len(p))[::-1]
    for yi, (_, r) in zip(y, p.iterrows()):
        clear = r.p_holm_primary_family < .05
        colour = COLOR['qmdp']
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=colour, linewidth=2, solid_capstyle='round')
        ax.plot(r.difference, yi, marker='o', markersize=8, color=colour,
                markerfacecolor=colour if clear else 'white', markeredgewidth=2)
        note = f'{r.difference:+.2f}' + ('  clearly worse' if clear and r.difference < 0 else
                                         '  clearly better' if clear else '')
        ax.text(1.02, yi, note, va='center', fontsize=8, color=INK if clear else INK2,
                fontweight='bold' if clear else 'normal', transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, [f"{SETTING_LABEL[r.setting].replace(chr(10), ' ')} · {r.seed_size} labels\nvs {NAMES[r.rival]}"
                      for _, r in p.iterrows()], fontsize=7.5)
    ax.set_xlabel('QMDP minus rival: average accuracy over 0–2,400 s (points), 95% CI')
    ax.set_xlim(min(p.ci_low.min(), -6) - .3, max(p.ci_high.max(), 1) + .4)
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[Line2D([], [], marker='o', color=COLOR['qmdp'], markerfacecolor=COLOR['qmdp'], linestyle='', label='Clear after Holm (12 tests)'),
                       Line2D([], [], marker='o', color=COLOR['qmdp'], markerfacecolor='white', markeredgewidth=2, linestyle='', label='Not clear')],
              loc='upper left', bbox_to_anchor=(0, -.11), ncol=2, frameon=False)
    ax.set_title('Final test, Stage 2: QMDP against the rival named in advance (20 seeds per cell)', loc='left')
    save(fig, 'fig1_stage2_primary_comparisons')


# ---------- Figure 2: accuracy curves ----------
def sample_curve(curve, budgets):
    """Same rule as src.eval.stage2.sample_curve: last accuracy measured at or before each time."""
    seconds = curve.seconds.to_numpy()
    accuracy = curve.accuracy_percent.to_numpy()
    index = np.searchsorted(seconds, np.asarray(budgets) + 1e-9, side='right') - 1
    return accuracy[index]


def load_curves(runs):
    cache = OUT / 'curves_on_common_time_grid.csv.gz'
    if cache.exists():
        return pd.read_csv(cache)
    grid = np.arange(0, 2401, 30)
    rows = []
    for _, run in runs.iterrows():
        curve = pd.read_csv(ROOT / run.run_folder / 'accuracy_curve.csv')
        values = sample_curve(curve, grid)
        rows.append(pd.DataFrame({'setting': run.setting, 'seed_size': run.seed_size, 'seed': run.seed,
                                  'policy': run.policy, 'seconds': grid, 'accuracy_percent': values}))
    curves = pd.concat(rows, ignore_index=True)
    curves.to_csv(cache, index=False)
    return curves


def fig_curves(runs):
    curves = load_curves(runs)
    for size in (50, 200):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), sharex=True)
        for ax, setting in zip(axes.flat, SETTINGS):
            subset = curves[(curves.setting == setting) & (curves.seed_size == size)]
            for policy in CURVE_POLICIES[::-1]:
                one = subset[subset.policy == policy].groupby('seconds').accuracy_percent
                mean, se = one.mean(), one.sem()
                ax.plot(mean.index, mean, color=COLOR[policy], linestyle=DASH[policy],
                        linewidth=2.4 if policy == 'qmdp' else 1.6, label=NAMES[policy], zorder=3 if policy == 'qmdp' else 2)
                if policy == 'qmdp':
                    ax.fill_between(mean.index, mean - 1.96 * se, mean + 1.96 * se, color=COLOR[policy], alpha=.15, linewidth=0)
            ax.set_title(SETTING_LABEL[setting].replace('\n', ' '), loc='left')
            ax.set_xticks([0, 600, 1200, 1800, 2400])
        for ax in axes[1]:
            ax.set_xlabel('Human seconds')
        for ax in axes[:, 0]:
            ax.set_ylabel('Test accuracy (%)')
        handles = [Line2D([], [], color=COLOR[p], linestyle=DASH[p], linewidth=2.4 if p == 'qmdp' else 1.6, label=NAMES[p])
                   for p in CURVE_POLICIES]
        fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, .03))
        fig.text(.01, .005, 'No fatigue: QMDP, Greedy, Greedy + rest and Ignores fatigue make identical choices, so their lines coincide.',
                 fontsize=7.5, color=INK2)
        fig.suptitle(f'Final test, Stage 2: classifier accuracy over time, {size} starting labels '
                     f'(mean of 20 seeds; band = 95% CI for QMDP)', x=.01, ha='left', fontsize=10)
        fig.tight_layout(rect=(0, .08, 1, .97))
        save(fig, f'fig2_stage2_accuracy_curves_{size}_labels')


# ---------- difference grids (Stage 2 and Stage 1) ----------
def difference_grid(table, rows, cols, row_labels, col_labels, title, name, fmt='{:+.2f}', figsize=(9, 5), note=''):
    fig, ax = plt.subplots(figsize=figsize)
    shade = {'ahead': AHEAD, 'behind': BEHIND, 'unclear': UNCLEAR}
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            cell = table.get((r, c))
            if cell is None:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='white', linewidth=2))
                continue
            value, state = cell
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=shade[state], edgecolor='white', linewidth=2))
            ax.text(j + .5, i + .5, fmt.format(value), ha='center', va='center', fontsize=7.2,
                    color=INK, fontweight='bold' if state != 'unclear' else 'normal')
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(cols)) + .5, col_labels, rotation=35, ha='right')
    ax.set_yticks(np.arange(len(rows)) + .5, row_labels)
    ax.tick_params(length=0); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=[Patch(facecolor=AHEAD, label='QMDP clearly ahead (Holm)'),
                       Patch(facecolor=UNCLEAR, label='Not clear'),
                       Patch(facecolor=BEHIND, label='QMDP clearly behind (Holm)')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False)
    ax.set_title(title, loc='left', pad=24)
    if note:
        fig.text(.01, -.02, note, fontsize=7.5, color=INK2, ha='left', va='top')
    save(fig, name)


def fig_stage2_grid():
    t = read(FINAL / 'stage2_tests.csv')
    t = t[t.metric == 'auc_accuracy_percent']
    rivals = ['rest_threshold', 'myopic', 'blind', 'two_choice', 'verify', 'full', 'random', 'random_image_full', 'qmdp_fixed']
    rows = [(s, n) for s in SETTINGS for n in (50, 200)]
    table = {((r.setting, r.seed_size), r.rival): (r.difference, status(r)) for _, r in t.iterrows()}
    difference_grid(table, rows, rivals,
                    [f"{SETTING_LABEL[s].replace(chr(10), ' ')} · {n}" for s, n in rows], [NAMES[r] for r in rivals],
                    'Final test, Stage 2: QMDP minus each rival, average accuracy 0–2,400 s (points)',
                    'fig3_stage2_all_rivals', figsize=(9.5, 6.2),
                    note='Rows: setting · starting labels. Holm correction per rival across the 12 setting × size cells; 20 paired seeds per cell.')


def fig_stage1_grid():
    t = read(FINAL / 'stage1_tests.csv')
    t['state'] = t.apply(status, axis=1)
    order = (['no_fatigue'] + [f'Markov_beta_{b}' for b in ('0.5', '1', '2', '3')] +
             [f'FALCON_beta_{b}' for b in ('0.5', '1', '2', '3')] + [f'Lapse_beta_{b}' for b in ('0.5', '1', '2', '3')] +
             [f'Lapse_seconds_beta_{b}' for b in ('0.5', '1', '2', '3')] + ['lapse_without_fatigue', 'lapse_seconds_without_fatigue'] +
             ['fast_stress_markov_beta_1', 'fast_stress_markov_beta_2'] +
             sorted(t[t.family == 'sensitivity'].case.unique()))
    label = {c: c.replace('_beta_', ', β = ').replace('Lapse_seconds', 'Lapse (seconds)').replace('_', ' ') for c in order}
    label = {c: v[0].upper() + v[1:] for c, v in label.items()}
    label.update({'no_fatigue': 'No fatigue', 'lapse_without_fatigue': 'Lapse, no fatigue (control)',
                  'lapse_seconds_without_fatigue': 'Lapse (seconds), no fatigue (control)',
                  'fast_stress_markov_beta_1': 'Fast stress Markov, β = 1', 'fast_stress_markov_beta_2': 'Fast stress Markov, β = 2',
                  'long_session_300': 'Long session (300 steps)'})
    rivals = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload', 'blind',
              'myopic', 'rest_threshold', 'qmdp_fixed', 'grid', 'dqn']
    table = {(r.case, r.rival): (r.difference, r.state) for _, r in t.iterrows()}
    difference_grid(table, order, rivals, [label[c] for c in order], [NAMES[r] for r in rivals],
                    'Final test, Stage 1: QMDP minus each policy, net reward (bits − λ·seconds, λ = 1)',
                    'fig4_stage1_all_cases', fmt='{:+.1f}', figsize=(10.5, 11),
                    note='The first eight columns are the required baselines B1–B8. 500 matched test sessions per case; '
                         'Holm correction per rival within each case family.')


# ---------- Figure 5: advantage against β ----------
def fig_beta():
    t = read(FINAL / 'stage1_tests.csv')
    families = [('Markov', 'Markov'), ('FALCON', 'FALCON'), ('Lapse', 'Lapse (onset in answers)'),
                ('Lapse_seconds', 'Lapse (onset in seconds)')]
    rivals = ['myopic', 'rest_threshold', 'verify', 'dqn']
    betas = [.5, 1, 2, 3]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=False)
    offsets = dict(zip(rivals, [-.09, -.03, .03, .09]))
    for ax, (prefix, title) in zip(axes, families):
        for rival in rivals:
            rows = [t[(t.case == f"{prefix}_beta_{'0.5' if b == .5 else int(b)}") & (t.rival == rival)].iloc[0] for b in betas]
            x = np.array(betas) + offsets[rival]
            d = np.array([r.difference for r in rows])
            lo = d - np.array([r.ci_low for r in rows]); hi = np.array([r.ci_high for r in rows]) - d
            ax.errorbar(x, d, yerr=[lo, hi], color=COLOR[rival], marker=MARKER[rival], markersize=6,
                        linestyle=DASH[rival], linewidth=1.6, capsize=0, label=NAMES[rival])
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(betas, ['0.5', '1', '2', '3'])
        ax.set_xlabel('Fatigue strength β')
        ax.set_title(title, loc='left')
    axes[0].set_ylabel('QMDP minus policy (net reward)')
    handles = [Line2D([], [], color=COLOR[r], marker=MARKER[r], linestyle=DASH[r], label=NAMES[r]) for r in rivals]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, -.06))
    fig.suptitle('Final test, Stage 1: does QMDP\'s lead grow with fatigue strength? (95% CI; above 0 = QMDP ahead)',
                 x=.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95))
    save(fig, 'fig5_stage1_advantage_by_beta')


# ---------- Figure 6: answers and label errors (mechanism) ----------
def fig_mechanism():
    s = read(FINAL / 'stage2_summary.csv')
    s = s[s.seed_size == 200]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    for ax, setting in zip(axes.flat, SETTINGS):
        sub = s[s.setting == setting].set_index('policy')
        for policy in CURVE_POLICIES:
            r = sub.loc[policy]
            ax.plot(r.acquired_answers, 100 * r.acquired_error_rate, marker=MARKER[policy], color=COLOR[policy],
                    markersize=9 if policy == 'qmdp' else 7, linestyle='', markeredgecolor='white', markeredgewidth=1.2,
                    label=NAMES[policy], zorder=3 if policy == 'qmdp' else 2)
        ax.set_title(SETTING_LABEL[setting].replace('\n', ' '), loc='left')
        ax.margins(.15)
    for ax in axes[1]:
        ax.set_xlabel('New answers acquired in 2,400 s')
    for ax in axes[:, 0]:
        ax.set_ylabel('Wrong acquired labels (%)')
    handles = [Line2D([], [], marker=MARKER[p], color=COLOR[p], linestyle='', markersize=7, label=NAMES[p]) for p in CURVE_POLICIES]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, .03))
    fig.text(.01, .005, 'No fatigue: QMDP, Greedy, Greedy + rest and Ignores fatigue coincide (one blue point).',
             fontsize=7.5, color=INK2)
    fig.suptitle('Final test, Stage 2: how many answers each policy gets, and how many are wrong (200 starting labels, mean of 20 seeds)',
                 x=.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .97))
    save(fig, 'fig6_stage2_answers_and_errors')


# ---------- Figure 7: full grid ----------
def fig_grid(report):
    rates = report['grid']['qmdp_vs_rivals']
    rivals = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload', 'blind',
              'myopic', 'rest_threshold', 'qmdp_fixed', 'grid', 'dqn']
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True)
    y = np.arange(len(rivals))[::-1]
    for ax, family in zip(axes, ['Markov', 'FALCON', 'Lapse']):
        ahead = np.array([rates[r][family]['clearly_ahead'] for r in rivals]) * 100
        behind = np.array([rates[r][family]['clearly_behind'] for r in rivals]) * 100
        ax.barh(y, ahead, color=COLOR['qmdp'], height=.7, label='QMDP clearly ahead')
        ax.barh(y, -behind, color='#e34948', height=.7, label='QMDP clearly behind')
        for yi, a, b in zip(y, ahead, behind):
            ax.text(a + 2, yi, f'{a:.0f}', va='center', fontsize=7, color=INK2)
            if b >= .5:
                ax.text(-b - 2, yi, f'{b:.0f}', va='center', ha='right', fontsize=7, color=INK2)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_xlim(-75, 118)
        ax.set_xticks([-50, 0, 50, 100], ['50', '0', '50', '100'])
        ax.set_xlabel('% of grid conditions')
        ax.set_title(family, loc='left')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(y, [NAMES[r] for r in rivals])
    h2 = report['grid']['h2_rate_beta_ge_1_unadjusted']
    fig.legend(handles=[Patch(facecolor=COLOR['qmdp'], label='QMDP clearly ahead (95% CI above 0)'),
                        Patch(facecolor='#e34948', label='QMDP clearly behind')],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, -.05))
    fig.suptitle('Final test, full Stage-1 grid (7,812 conditions; descriptive, no correction). '
                 f"Beats all of B1–B8 at β ≥ 1: Markov {h2['Markov']}, FALCON {h2['FALCON']}, Lapse {h2['Lapse']}",
                 x=.01, ha='left', fontsize=9.5)
    fig.tight_layout(rect=(0, .05, 1, .95))
    save(fig, 'fig7_full_grid_rates')


# ---------- Figure 8: development vs final test ----------
def fig_dev_vs_test():
    rivals = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload', 'blind',
              'myopic', 'rest_threshold', 'qmdp_fixed', 'grid', 'dqn']
    test = read(FINAL / 'stage1_summary.csv').pivot_table(index='case', columns='policy', values='reward')
    dev = read(DEV_STAGE1).pivot_table(index='case', columns='policy', values='reward')
    cases = test.index.intersection(dev.index)
    pairs = []
    for c in cases:
        for r in rivals:
            if r in dev.columns and r in test.columns and not np.isnan(dev.loc[c, r]):
                pairs.append((dev.loc[c, 'qmdp'] - dev.loc[c, r], test.loc[c, 'qmdp'] - test.loc[c, r]))
    s1 = np.array(pairs)

    runs = read(FINAL / 'stage2_runs.csv')
    tmean = runs.groupby(['setting', 'seed_size', 'policy']).auc_accuracy_percent.mean().unstack('policy')
    d = read(DEV_STAGE2)
    d['setting'] = d.model + d.beta.astype(float).astype(int).astype(str)
    d.loc[d.model == 'Lapse', 'setting'] = 'Lapse1:forced'
    dmean = d.groupby(['setting', 'seed_size', 'policy']).auc_accuracy_percent.mean().unstack('policy')
    keys = tmean.index.intersection(dmean.index)
    s2 = np.array([(dmean.loc[k, 'qmdp'] - dmean.loc[k, p], tmean.loc[k, 'qmdp'] - tmean.loc[k, p])
                   for k in keys for p in tmean.columns if p != 'qmdp'])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, data, title, unit in [(axes[0], s1, f'Stage 1: {len(s1)} case × policy differences', 'net reward'),
                                  (axes[1], s2, f'Stage 2: {len(s2)} setting × size × policy differences', 'AUC points')]:
        low, high = data.min(), data.max()
        pad = .05 * (high - low)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED, linewidth=1, linestyle='--', zorder=1)
        ax.plot(data[:, 0], data[:, 1], 'o', color=COLOR['qmdp'], markersize=5, alpha=.75, markeredgecolor='white',
                markeredgewidth=.6, zorder=2)
        r = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        mad = np.median(np.abs(data[:, 1] - data[:, 0]))
        ax.text(.03, .97, f'r = {r:.3f}\nmedian |test − dev| = {mad:.2f}', transform=ax.transAxes, va='top', fontsize=8, color=INK2)
        ax.set_xlabel(f'Development: QMDP minus policy ({unit})')
        ax.set_ylabel(f'Final test: QMDP minus policy ({unit})')
        ax.set_title(title, loc='left')
        if unit == 'net reward':
            ax.set_xscale('symlog', linthresh=50); ax.set_yscale('symlog', linthresh=50)
            ticks = [-100, 0, 100, 1000]
            ax.set_xticks(ticks, [f'{t:,}' for t in ticks]); ax.set_yticks(ticks, [f'{t:,}' for t in ticks])
            ax.minorticks_off()
            ax.text(.97, .03, 'symmetric log axes (linear within ±50)', transform=ax.transAxes, ha='right', fontsize=7, color=INK2)
    fig.suptitle('Development results against final-test results (dashed line = identical)', x=.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, 'fig8_development_vs_final_test')
    return len(s1), len(s2)


# ---------- tables for the report ----------
def tables(report):
    lines = ['# Final-test tables', '', 'Made by scripts/make_final_test_figures.py from outputs/final_test.', '']
    p = read(FINAL / 'stage2_primary.csv')
    lines += ['## Table 1. Stage 2, the 12 main comparisons (QMDP minus rival, average accuracy 0–2,400 s, points)', '',
              '| Setting | Starting labels | Rival | Difference | 95% CI | p | p (Holm, 12 tests) | Result |',
              '|---|---|---|---|---|---|---|---|']
    for s in SETTINGS:
        for n in (50, 200):
            r = p[(p.setting == s) & (p.seed_size == n)].iloc[0]
            res = ('clearly better' if r.difference > 0 else 'clearly worse') if r.p_holm_primary_family < .05 else 'not clear'
            lines.append(f"| {SETTING_LABEL[s].replace(chr(10), ' ')} | {n} | {NAMES[r.rival]} | {r.difference:+.2f} | "
                         f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}] | {r.p:.4f} | {r.p_holm_primary_family:.4f} | {res} |")
    st = report['stage1']
    lines += ['', '## Table 2. Hypotheses on the final test', '', '| Hypothesis | Result | Evidence |', '|---|---|---|',
              f"| H1: no fatigue, QMDP within 1 SE of the best required baseline | holds | difference {st['h1']['qmdp_minus_best']:.2f} "
              f"(SE {st['h1']['se']:.2f}) against {NAMES[st['h1']['best_required_baseline']]} |",
              f"| H2: beats all of B1–B8 at β ≥ 1 (Holm) | Markov holds; FALCON partial; Lapse fails | Markov "
              f"{st['h2_markov_falcon_beta_ge_1_holm']['Markov']}, FALCON {st['h2_markov_falcon_beta_ge_1_holm']['FALCON']}, "
              f"core Lapse {st['all_cases_beating_all_B1_B8_holm_by_family']['core_Lapse']}, Lapse (seconds) "
              f"{st['all_cases_beating_all_B1_B8_holm_by_family']['lapse_seconds']} |",
              f"| H3: lead over greedy grows with β (Markov) | fails | gaps {', '.join(f'{g:.1f}' for g in st['h3_markov']['qmdp_minus_greedy_beta_0.5_1_2_3'])}; "
              f"slope {st['h3_markov']['slope_per_beta']:.2f} per β [{st['h3_markov']['slope_ci'][0]:.2f}, {st['h3_markov']['slope_ci'][1]:.2f}] |",
              f"| Robustness: beats greedy under misspecified fatigue (Holm) | {st['robustness_beats_greedy_holm']} | fails in the FALCON world |",
              f"| Stage 2: better than the named rival (Holm, 12 tests) | not shown | {report['stage2']['qmdp_clearly_better_after_holm']} clearly better, "
              f"{report['stage2']['qmdp_clearly_worse_after_holm']} clearly worse |"]
    s = read(FINAL / 'stage2_summary.csv')
    lines += ['', '## Table 3. Stage 2 means, 200 starting labels', '',
              '| Setting | Policy | Average accuracy | Accuracy at 2,400 s | Answers | Wrong labels |', '|---|---|---|---|---|---|']
    for setting in SETTINGS:
        for policy in CURVE_POLICIES:
            r = s[(s.setting == setting) & (s.seed_size == 200) & (s.policy == policy)].iloc[0]
            lines.append(f"| {SETTING_LABEL[setting].replace(chr(10), ' ')} | {NAMES[policy]} | {r.auc_accuracy_percent:.2f} | "
                         f"{r.accuracy_2400s:.2f} | {r.acquired_answers:.0f} | {100 * r.acquired_error_rate:.1f}% |")
    (OUT / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('wrote tables.md')


def main():
    OUT.mkdir(exist_ok=True)
    report = check_inputs()
    runs = read(FINAL / 'stage2_runs.csv')
    assert len(runs) == 2400 and runs.final_test_evaluated.astype(str).eq('True').all()
    fig_primary()
    fig_curves(runs)
    fig_stage2_grid()
    fig_stage1_grid()
    fig_beta()
    fig_mechanism()
    fig_grid(report)
    print('development vs test pairs:', fig_dev_vs_test())
    tables(report)


if __name__ == '__main__':
    main()
