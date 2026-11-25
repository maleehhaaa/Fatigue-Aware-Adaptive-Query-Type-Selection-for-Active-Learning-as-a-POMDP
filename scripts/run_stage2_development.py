"""Run or resume the default Stage-2 development study.

    python scripts/run_stage2_development.py                 # main study (what notebook 10 runs)
    python scripts/run_stage2_development.py --lambda-check  # review item 13: lambda 0.1/0.25/0.5
    python scripts/run_stage2_development.py --workers 12

Development data only; the final test images are never evaluated here.
"""
from pathlib import Path
import argparse
import sys
import json

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))

from src.eval.stage2 import run_stage2_study, comparison_tables, run_lambda_check


def main_study(workers):
    table, folder = run_stage2_study(workers=workers)
    accuracy, target_time = comparison_tables(table)
    accuracy.to_csv(folder / 'paired_accuracy_comparisons.csv', index=False)
    # Time to 80%: only pairs where both runs reached it (review fix 12); never capped times.
    target_time.to_csv(folder / 'paired_target_time_comparisons.csv', index=False)
    pointer = {'folder': str(folder.relative_to(project)), 'runs': len(table),
        'policies': sorted(table.policy.unique()),
        'status': 'preliminary Stage-2 development study (simulator after the 19 Nov 2026 review)',
        'final_test_evaluated': False}
    (project / 'outputs/stage2_latest.json').write_text(json.dumps(pointer, indent=2))
    print(json.dumps(pointer, indent=2), flush=True)


def lambda_check(workers):
    table = run_lambda_check(workers=workers)
    folder = project / 'outputs/stage2_lambda_check'
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / 'lambda_check_runs.csv', index=False)
    final = max((c for c in table if c.startswith('accuracy_') and c.endswith('s')),
        key=lambda c: float(c[len('accuracy_'):-1]))
    summary = (table.groupby(['model', 'beta', 'seed_size', 'policy', 'lam'])
        [['auc_accuracy_percent', final]].mean().round(2))
    summary.to_csv(folder / 'lambda_check_summary.csv')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=6, help='parallel CPU workers (default 6)')
    parser.add_argument('--lambda-check', action='store_true', help='run the lambda check instead of the main study')
    arguments = parser.parse_args()
    if arguments.lambda_check:
        lambda_check(arguments.workers)
    else:
        main_study(arguments.workers)
