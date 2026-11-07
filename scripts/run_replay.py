"""Run the taxi replay. Validation is the default; test is one-time only.

Examples:
  python scripts/run_replay.py --split val --max-days 1 --max-decisions 5
  python scripts/run_replay.py --split val
  python scripts/run_replay.py --split test --confirm
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'src'))

from taxi import config  # noqa: E402
from taxi.replay.runner import run_validation  # noqa: E402


def check_test_guard(split, confirm, max_days, max_decisions, lock):
    """Validate before touching any test data; no --force escape hatch."""
    if split != 'test':
        return
    if not confirm:
        raise ValueError('Test replay is one-time. Freeze the plan and add --confirm.')
    if max_days is not None or max_decisions is not None:
        raise ValueError('Partial test runs are forbidden. Use --split val for debugging.')
    if lock.exists():
        raise FileExistsError(f'Test replay already completed: {lock}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--confirm', action='store_true', help='Confirm one-time frozen test evaluation')
    p.add_argument('--max-days', type=int, default=None, help='Validation debugging only')
    p.add_argument('--max-decisions', type=int, default=None, help='Validation debugging only')
    p.add_argument('--speed-kmh', type=float, default=25.0,
                   help='Approximate road speed; agree/freeze on validation')
    p.add_argument('--empty-tolerance-min', type=float, default=10.0,
                   help='Comparison minutes-empty tolerance; freeze on validation')
    p.add_argument('--min-episodes', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(argv)
    lock = config.OUT_DIR / 'replay' / 'test' / 'TEST_RUN_DONE'
    check_test_guard(a.split, a.confirm, a.max_days, a.max_decisions, lock)
    if a.min_episodes < 5:
        p.error('min-episodes must be >= 5')
    if a.empty_tolerance_min < 0:
        p.error('empty-tolerance-min must be >= 0')
    out, count = run_validation(split=a.split, max_days=a.max_days,
                                max_decisions=a.max_decisions, speed_kmh=a.speed_kmh,
                                empty_tolerance_min=a.empty_tolerance_min,
                                min_episodes=a.min_episodes, seed=a.seed)
    if a.split == 'test':
        # Write lock only after all outputs were successfully written.
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('x') as f:
            f.write(f'Completed {datetime.now().isoformat()} with {count} decisions\n')
    print(f'Completed {count} decisions; results saved to {out}')
    if a.max_days is not None or a.max_decisions is not None:
        print('DEBUG RUN ONLY: not a full evaluation.')


if __name__ == '__main__':
    main()
