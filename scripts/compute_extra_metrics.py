"""Extra metrics named in the project description, computed from the saved final-test files.

No experiment is run and no test data is used again: this reads outputs/final_test/stage1_sessions.csv.gz
and the saved answer_audit.csv.gz of each of the 2,400 final-test Stage-2 runs.

These metrics were defined AFTER the final test was run, so they are exploratory and descriptive:
95% paired bootstrap intervals are shown, but no p-values and no claims of significance.

Definitions
Stage 2 (from the per-call audit; true hidden fatigue F is on the 0-4 scale; action 0 = pause,
1 = verify, 2 = two-choice, 3 = full question; a call cut off by the end of a 600-s shift is censored):
  interactions              questions asked and answered (actions 1-3, not censored)
  pauses, rest share        pauses taken, and the share of the 2,400 human seconds spent pausing
  unnecessary interruptions questions asked while the person could not answer reliably:
                            true fatigue F >= 3 ("highly fatigued"), or after the person lapsed
                            (Lapse worlds only). Reported as a count per run and as a share of interactions.
  wrong answers             interactions whose answer was wrong (label noise), count per run
  burden: mean fatigue      time-weighted mean of the true fatigue level over the human seconds used
  burden: time at F >= 3    share of the human seconds spent at true fatigue F >= 3
  gain per 100 answers      (accuracy at 2,400 s - starting accuracy) / interactions * 100.
                            Because of diminishing returns this ratio favours policies that collect fewer
                            answers, so it is read together with the accuracy results, not alone.
Stage 1 (from the per-session totals; true fatigue per step was not saved, so no burden measure here):
  interactions              opportunities used to ask (150 - pauses); gold checks count as interactions
  bits per interaction      information (bits) / interactions
  bits per human second     information / seconds (information per unit of human effort)

Usage:  python scripts/compute_extra_metrics.py
Output: outputs/final_test/extra_metrics/
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'outputs' / 'final_test'
OUT = FINAL / 'extra_metrics'
HIGH = 3.0
B, SEED = 4000, 42
SETTINGS = ['None0', 'Markov1', 'Markov2', 'FALCON1', 'Lapse1:forced', 'Lapse1:forced_seconds']
STAGE2_RIVALS = ['rest_threshold', 'myopic', 'blind', 'two_choice', 'verify', 'full', 'random', 'random_image_full', 'qmdp_fixed']
STAGE1_RIVALS = ['full', 'two_choice', 'verify', 'random', 'alternating', 'exp_decay', 'gap_workload', 'blind',
                 'myopic', 'rest_threshold', 'qmdp_fixed', 'grid', 'dqn']
STAGE2_METRICS = ['interactions', 'pauses', 'rest_share', 'unnecessary', 'unnecessary_share', 'unnecessary_high_fatigue',
                  'unnecessary_lapsed', 'wrong_answers', 'mean_fatigue', 'time_high_fatigue', 'gain_per_100_answers']
STAGE1_METRICS = ['interactions', 'bits_per_interaction', 'bits_per_second']


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def paired_ci(diff, rng):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return np.nan, np.nan, np.nan
    if np.allclose(diff, diff[0]):
        return diff.mean(), diff[0], diff[0]
    means = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
    return diff.mean(), *np.percentile(means, [2.5, 97.5])


def stage2_run_metrics(run):
    a = pd.read_csv(ROOT / run.run_folder / 'answer_audit.csv.gz')
    duration = (a.seconds - a.start_seconds).to_numpy()
    asked = (a.accepted & (a.action > 0)).to_numpy()
    high = (a.fatigue >= HIGH - 1e-9).to_numpy()
    lapsed = a.lapsed.astype(bool).to_numpy()
    n = asked.sum()
    used = duration.sum()
    return {
        'interactions': int(n),
        'pauses': int((a.action == 0).sum()),
        'rest_share': float(duration[(a.action == 0).to_numpy()].sum() / used) if used else np.nan,
        'unnecessary': int((asked & (high | lapsed)).sum()),
        'unnecessary_share': float((asked & (high | lapsed)).sum() / n) if n else np.nan,
        'unnecessary_high_fatigue': int((asked & high).sum()),
        'unnecessary_lapsed': int((asked & lapsed).sum()),
        'wrong_answers': int((asked & ~a.correct.astype(bool).to_numpy()).sum()),
        'mean_fatigue': float((a.fatigue.to_numpy() * duration).sum() / used) if used else np.nan,
        'time_high_fatigue': float(duration[high].sum() / used) if used else np.nan,
        'human_seconds': float(used),
        'gain_per_100_answers': float((run.accuracy_2400s - run.initial_accuracy) / n * 100) if n else np.nan,
    }


def stage2():
    runs = read(FINAL / 'stage2_runs.csv')
    assert len(runs) == 2400 and set(runs.evaluation_images) == {'final_test'}
    rows = []
    for _, run in runs.iterrows():
        rows.append({'setting': run.setting, 'seed_size': run.seed_size, 'seed': run.seed, 'policy': run.policy,
                     'acquired_answers': run.acquired_answers, **stage2_run_metrics(run)})
    table = pd.DataFrame(rows)
    # Consistency check against the saved run summaries.
    assert (table.interactions == table.acquired_answers).all(), 'interaction count differs from acquired_answers'
    table.to_csv(OUT / 'stage2_run_metrics.csv', index=False)
    summary = table.groupby(['setting', 'seed_size', 'policy'])[STAGE2_METRICS].mean().reset_index()
    summary.to_csv(OUT / 'stage2_summary.csv', index=False)
    rng = np.random.default_rng(SEED)
    comparisons = []
    for (setting, size), group in table.groupby(['setting', 'seed_size']):
        wide = {m: group.pivot(index='seed', columns='policy', values=m) for m in STAGE2_METRICS}
        for rival in STAGE2_RIVALS:
            for m in STAGE2_METRICS:
                d, lo, hi = paired_ci(wide[m]['qmdp'] - wide[m][rival], rng)
                comparisons.append({'setting': setting, 'seed_size': size, 'rival': rival, 'metric': m,
                                    'qmdp_minus_rival': d, 'ci_low': lo, 'ci_high': hi})
    pd.DataFrame(comparisons).to_csv(OUT / 'stage2_qmdp_vs_rivals.csv', index=False)
    return table, summary


def stage1():
    s = pd.read_csv(FINAL / 'stage1_sessions.csv.gz', keep_default_na=False, na_values=[''])
    steps = s[['count_skip', 'count_verify', 'count_two_choice', 'count_full', 'count_gold']].sum(1)
    s['interactions'] = steps - s.count_skip
    s['bits_per_interaction'] = s.information / s.interactions.replace(0, np.nan)
    s['bits_per_second'] = s.information / s.seconds
    summary = s.groupby(['family', 'case', 'policy'])[STAGE1_METRICS + ['count_skip']].mean().reset_index()
    summary.to_csv(OUT / 'stage1_summary.csv', index=False)
    rng = np.random.default_rng(SEED)
    comparisons = []
    for case, group in s.groupby('case'):
        wide = {m: group.pivot(index='episode', columns='policy', values=m) for m in STAGE1_METRICS}
        for rival in STAGE1_RIVALS:
            for m in STAGE1_METRICS:
                d, lo, hi = paired_ci(wide[m]['qmdp'] - wide[m][rival], rng)
                comparisons.append({'case': case, 'rival': rival, 'metric': m, 'qmdp_minus_rival': d, 'ci_low': lo, 'ci_high': hi})
    pd.DataFrame(comparisons).to_csv(OUT / 'stage1_qmdp_vs_rivals.csv', index=False)
    return summary


def figure(summary):
    import importlib.util
    spec = importlib.util.spec_from_file_location('final_figures', ROOT / 'scripts' / 'make_final_test_figures.py')
    figs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs)
    plt = figs.plt
    policies = figs.CURVE_POLICIES
    panels = [('unnecessary', 'Unnecessary interruptions per run\n(asked at F ≥ 3 or after a lapse)'),
              ('mean_fatigue', 'Burden: time-weighted mean\ntrue fatigue (0–4)'),
              ('time_high_fatigue', 'Burden: share of human time\nat F ≥ 3 (%)'),
              ('gain_per_100_answers', 'Accuracy gain per 100 answers\n(points)')]
    sub = summary[summary.seed_size == 200]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.2), sharey=True)
    y = np.arange(len(SETTINGS))[::-1]
    offsets = np.linspace(-.3, .3, len(policies))
    for ax, (metric, title) in zip(axes, panels):
        for off, policy in zip(offsets, policies):
            values = [sub[(sub.setting == s) & (sub.policy == policy)][metric].iloc[0] for s in SETTINGS]
            values = np.array(values) * (100 if metric == 'time_high_fatigue' else 1)
            ax.plot(values, y + off, linestyle='', marker=figs.MARKER[policy], color=figs.COLOR[policy],
                    markersize=8 if policy == 'qmdp' else 6, markeredgecolor='white', markeredgewidth=.8,
                    label=figs.NAMES[policy], zorder=3 if policy == 'qmdp' else 2)
        ax.set_title(title, loc='left', fontsize=9)
        ax.grid(axis='y', visible=False)
        for yi in y[:-1]:
            ax.axhline(yi - .5, color=figs.GRID, linewidth=.8)
    axes[0].set_yticks(y, [figs.SETTING_LABEL[s].replace('\n', ' ') for s in SETTINGS])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, -.02))
    fig.suptitle('Final test, Stage 2 (200 starting labels, mean of 20 seeds): interruptions, burden and gain per answer '
                 '— exploratory metrics defined after the test', x=.01, ha='left', fontsize=9.5)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(OUT / 'fig9_interruptions_and_burden.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUT / 'fig9_interruptions_and_burden.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote fig9_interruptions_and_burden')


def main():
    OUT.mkdir(exist_ok=True)
    for part in ['stage1', 'stage2']:
        assert (FINAL / f'{part}_complete.json').exists()
    _, summary2 = stage2()
    stage1()
    figure(summary2)
    print('wrote', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
