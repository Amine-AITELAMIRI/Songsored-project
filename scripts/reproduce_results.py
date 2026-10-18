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


def classification_summary(name, models, expected_n):
    rows = load(name)
    assert len(rows) == len(models) * len(SEEDS)
    result = {}
    for model in models:
        group = [r for r in rows if r['model'] == model]
        assert len(group) == 5 and {int(r['seed']) for r in group} == SEEDS
        accuracy, losses = [], []
        for row in group:
            n, tp, tn, fp, fn = (int(row[k]) for k in ('n', 'tp', 'tn', 'fp', 'fn'))
            assert n == expected_n and tp + tn + fp + fn == n
            assert min(tp, tn, fp, fn) >= 0
            assert 0 < int(row['selected_step']) <= 24600
            assert int(row['selected_step']) % 82 == 0
            accuracy.append((tp + tn) / n)
            losses.append(float(row['bce']))
        result[model] = dict(accuracy_mean=mean(accuracy),
                             accuracy_seed_sd=stdev(accuracy), bce_mean=mean(losses))
    return result


def compute():
    rows = load('position_counts.csv')
    models = ['global_mean', 'ordered', 'ordered_jitter']
    offsets = [-200, -100, 0, 100, 200]
    assert len(rows) == 75
    curves = {}
    for model in models:
        curve = {}
        for offset in offsets:
            group = [r for r in rows if r['model'] == model and int(r['offset_ms']) == offset]
            assert len(group) == 5 and {int(r['seed']) for r in group} == SEEDS
            assert all(int(r['positive_clips']) == 25 and 0 <= int(r['detected']) <= 25 for r in group)
            rates = [int(r['detected']) / int(r['positive_clips']) for r in group]
            curve[str(offset)] = dict(recall_mean=mean(rates), recall_seed_sd=stdev(rates))
        curves[model] = curve
    return dict(
        representation=classification_summary('representation_metrics.csv', ['cnn', 'frozen_mean'], 41),
        centered_development=classification_summary('centered_metrics.csv', models, 70),
        position=curves,
        nonzero_offset_recall={m: mean(v['recall_mean'] for k, v in curve.items() if k != '0') for m, curve in curves.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Compare recomputed values with the committed summary.')
    args = parser.parse_args()
    early_history()
    result = compute()
    if args.check:
        expected = json.loads((ROOT / 'evidence' / 'summary.json').read_text())
        assert result == expected, 'Recomputed results differ from the committed summary.'
        print('PASS: counts and aggregates reproduce; early loss histories are complete and finite.')
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
