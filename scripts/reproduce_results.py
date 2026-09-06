#!/usr/bin/env python3
"""Recompute reported aggregates from sanitized per-run counts (no audio needed)."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {42, 7, 21, 123, 2026}


def load(name):
    with (ROOT / 'evidence' / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def early_history():
    """Validate the aggregate loss history used by the early CNN figure."""
    rows = load('early_cnn_history.csv')
    assert len(rows) == 2000
    assert {r['condition'] for r in rows} == {'raw100', 'norm300'}
    for condition, epochs in [('raw100', 100), ('norm300', 300)]:
        group = [r for r in rows if r['condition'] == condition]
        assert {int(r['seed']) for r in group} == SEEDS
        for seed in SEEDS:
            run = [r for r in group if int(r['seed']) == seed]
            assert len(run) == epochs
            assert sorted(int(r['epoch']) for r in run) == list(range(1, epochs + 1))
            assert all(math.isfinite(float(r[k])) and float(r[k]) >= 0
                       for r in run for k in ('train_bce', 'val_bce'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate loss-history coverage and values.')
    parser.parse_args()
    early_history()
    print('PASS: early loss histories are complete and finite.')


if __name__ == '__main__':
    main()
