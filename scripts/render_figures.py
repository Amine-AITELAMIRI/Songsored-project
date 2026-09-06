#!/usr/bin/env python3
"""Redraw the project figures from aggregate counts and loss histories."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statistics import mean, stdev
from reproduce_results import early_history

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})
history = early_history()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained')
for condition, label, color, epochs in [
    ('raw100', 'Raw inputs', '#245c96', 100),
    ('norm300', 'Normalized inputs', '#20806c', 300),
]:
    xs = list(range(1, epochs + 1))
    for ax, field, title in zip(axes, ['train_bce', 'val_bce'], ['Training', 'Validation']):
        values = [[float(r[field]) for r in history
                   if r['condition'] == condition and int(r['epoch']) == epoch] for epoch in xs]
        means = [mean(v) for v in values]
        sds = [stdev(v) for v in values]
        ax.plot(xs, means, label=label, color=color, linewidth=1.7)
        ax.fill_between(xs, [m - s for m, s in zip(means, sds)],
                        [m + s for m, s in zip(means, sds)], color=color, alpha=.12)
        ax.set(title=title, xlabel='Epoch', ylabel='Binary cross-entropy')
for ax in axes:
    ax.axvline(100, color='#777777', linestyle=':', linewidth=1.2)
    ax.grid(axis='y', alpha=.18)
axes[0].legend(frameon=False)
fig.suptitle('Early CNN runs: normalization helps, longer training leaves a gap', fontsize=12)
fig.savefig(ROOT / 'assets' / 'cnn_learning_curves.png', dpi=170, bbox_inches='tight',
            metadata={'Software': 'Matplotlib'})
plt.close(fig)

print('Rendered figures from aggregate evidence.')
