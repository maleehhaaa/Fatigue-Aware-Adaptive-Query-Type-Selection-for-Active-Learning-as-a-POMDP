# Fatigue-Aware Adaptive Query-Type Selection for Active Learning

Course project for MLR570 (Advanced Machine Learning), American University of Sharjah.
Maleeha Fatima, Sara Alraeesi, Shaikha Alzaabi.

An active learner asks a human about uncertain images. Each time, a controller decides whether to
**pause** (let the person rest) or which question to ask: **verify** ("Is this a cat?"),
**two-choice** ("cat, dog or neither?") or **full label**. The person's fatigue is hidden, so the
controller tracks it from response times and plans ahead (a POMDP solved with QMDP). The objective
is information about the true labels per second of human time. People are simulated, calibrated
to [CIFAR-10H](https://github.com/jcpeterson/cifar-10h); the fatigue models are assumptions.

## Main results (final test, held-out people and images)

- Under recoverable (Markov) fatigue, QMDP gets more information per human second than all eight
  required baselines in 16 of 17 cases (Holm-corrected), and than a greedy controller.
- Its lead over greedy does not grow with fatigue strength, and it is fragile when its fatigue
  model is wrong (FALCON 1/3 cases; disengagement 0/8).
- With a real classifier retrained from the answers, QMDP clearly beats ignoring fatigue, but a
  tuned greedy-plus-rest rule matches it: the benefit comes from *when to pause*, not *how to ask*.

Details are in the report and in `notebooks/11_final_test_results.ipynb`.

## Repository layout

```
configs/    fixed settings, including frozen_src.json (fingerprint of src/)
src/        sim/ (simulated people), control/ (belief, QMDP, grid POMDP, DQN, baselines),
            learn/ (classifier, partial-label loss, learning loop), eval/ (stages, studies, stats)
notebooks/  00-11, the full pipeline in order
scripts/    lambda sweep, full grid, development checks (E3, E4, E5, break-even, Lapse checks),
            final test, figures, exploratory metrics
tests/      34 unit tests
data/       splits and processed files (created by notebooks 01-03)
outputs/    every run, saved as outputs/<stage>/<config-hash>/seed-<k>/ with its config.json
```

## Setup

Python 3.12. Results were produced on macOS (Apple silicon); other platforms may differ slightly.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notebook 01 downloads CIFAR-10H, and notebook 02 downloads CIFAR-10 and the ImageNet ResNet-18
weights. These raw downloads are not included in this repository.

## Not included

To stay within GitHub's size limits, the repository leaves out the raw downloads
(`data/cifar10/`, `data/cifar10h/`, `data/pretrained/`) and the run caches
`outputs/stage1/`, `outputs/stage2_development/` and the full grid's
`candidate_sessions.npz` (about 6.9 GB). All other outputs are included, so every number in the
report can be checked against a saved file. Running the notebooks and scripts downloads the data
again and rebuilds the caches. `run_full_grid.py summarise` reads `candidate_sessions.npz`, so it
needs the `run` part first; the saved selection is in `selection.json` and `candidate_scores.csv`.

## Reproducing the results

1. Run `notebooks/00` to `10` in order. Each run is cached by its configuration and code
   fingerprint, so a rerun with unchanged code reuses saved results. The first run rebuilds the
   caches left out of the repository, which takes longer.
2. Development checks and controller selection (development data only). Each script runs in
   parts; `a|b` means run part `a`, then part `b`:
   ```bash
   python scripts/run_lambda_sweep.py theory|stage2|analyse
   python scripts/run_full_grid.py run --workers 8     # selection on 7,812 conditions
   python scripts/run_e3_ablation.py stage1|stage2|analyse
   python scripts/run_e4_headroom.py
   python scripts/run_e5_pause_length.py stage1|stage2|analyse
   python scripts/pause_break_even.py theory|observed
   python scripts/run_lapse_seconds_check.py stage1|stage2|analyse
   python scripts/run_lapse_lambda_check.py stage1|stage2|analyse
   ```
   Most scripts also have a `smoke` part for a quick check; `--workers` sets the number of
   parallel processes.
3. The final test (`scripts/run_final_test.py`) uses held-out test people and images. Its
   settings are fixed in `outputs/final_test/settings.json`, and each part runs only once.
   `notebooks/11_final_test_results.ipynb` reads the saved results;
   `scripts/make_final_test_figures.py` and `scripts/compute_extra_metrics.py` rebuild the figures
   and the exploratory metrics.

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

`src/` is frozen (`configs/frozen_src.json`); the full-grid, final-test and E1b scripts refuse to
run on changed code.
