"""Standalone cross-memory capacity (CMC) experiment for the fixed PT reservoir.

Place in src/run_cross_memory_capacity.py and run from project root:
    python3 -m src.run_cross_memory_capacity

Uses the existing src.tasks.properties.cross_memory.evaluate_cross_memory.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task
from src.tasks.properties.cross_memory import evaluate_cross_memory
from types import SimpleNamespace

# Same fixed-circuit and input protocol as the validated LMC experiment.
FC = 48e6
TS_RATIO = 0.5
SYMBOL_PERIOD = TS_RATIO / FC
ETA = 1.0
ALPHA = 8.33
INPUT_SCALE = 1e-4
N_SYMBOLS = 3000
WASHOUT = 200
TRAIN_RATIO = 0.8
POINTS_PER_SYMBOL = 16
K = 8
MAX_DELAY = 20
RIDGE = 1e-6
NORMALIZE = True
SEED = 42
ENCODING = 'zoh'
SOLVER = 'exact_zoh'
OUTPUT = Path('results/cross_memory_capacity')


def main():
    rng = np.random.default_rng(SEED)
    u = rng.choice([-1.0, 1.0], size=N_SYMBOLS)
    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    task = SimpleNamespace(u=u, y=np.zeros_like(u))

    sim = simulate_continuous_task(
        reservoir=reservoir,
        task_data=task,
        symbol_period=SYMBOL_PERIOD,
        points_per_symbol=POINTS_PER_SYMBOL,
        k=K,
        amplitude_scale=INPUT_SCALE,
        encoding=ENCODING,
        solver=SOLVER,
    )
    X = np.asarray(sim.X)
    if X.shape != (N_SYMBOLS, 2 * K):
        raise ValueError(f'Unexpected features shape: {X.shape}')

    pairs = [(i, j) for i in range(1, MAX_DELAY + 1)
             for j in range(i + 1, MAX_DELAY + 1)]
    result = evaluate_cross_memory(
        X=X, u=u, pairs=pairs, max_delay=MAX_DELAY,
        washout=WASHOUT, train_ratio=TRAIN_RATIO,
        ridge=RIDGE, normalize=NORMALIZE,
    )
    train_scores = np.asarray(result.train_scores, dtype=float)
    test_scores = np.asarray(result.test_scores, dtype=float)
    if len(train_scores) != len(pairs) or len(test_scores) != len(pairs):
        raise ValueError('Number of returned scores differs from number of pairs')

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'cross_memory_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['delay_1', 'delay_2', 'train_cmc', 'test_cmc'])
        for (i, j), tr, te in zip(pairs, train_scores, test_scores):
            writer.writerow([i, j, f'{tr:.12g}', f'{te:.12g}'])

    heatmap = np.full((MAX_DELAY, MAX_DELAY), np.nan)
    for (i, j), score in zip(pairs, test_scores):
        heatmap[j - 1, i - 1] = score
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(heatmap, origin='lower', interpolation='nearest',
                   extent=(0.5, MAX_DELAY + 0.5, 0.5, MAX_DELAY + 0.5),
                   aspect='equal')
    ax.set(xlabel='First delay $d_1$ (symbols)',
           ylabel='Second delay $d_2$ (symbols)',
           title=rf'PT Single: Cross Memory Capacity ($\eta$={ETA}, $\alpha$={ALPHA})')
    fig.colorbar(im, ax=ax, label='Test CMC')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'cross_memory_heatmap.png', dpi=200)
    plt.close(fig)

    separations = np.array([j - i for i, j in pairs])
    gaps = np.arange(1, MAX_DELAY)
    means = np.array([test_scores[separations == g].mean() for g in gaps])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gaps, means, 'o-')
    ax.set(xlabel='Delay separation $d_2-d_1$ (symbols)',
           ylabel='Mean test CMC', title='Cross Memory vs. Delay Separation')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'cross_memory_separation.png', dpi=200)
    plt.close(fig)

    best = int(np.argmax(test_scores))
    n_valid = N_SYMBOLS - max(WASHOUT, MAX_DELAY)
    n_train = int(n_valid * TRAIN_RATIO)
    print('\n=== Cross Memory Capacity ===')
    print(f'Circuit           : eta={ETA}, alpha={ALPHA}')
    print(f'Input             : bipolar +/-1, seed={SEED}, scale={INPUT_SCALE:g} A')
    print(f'Ts/Tc             : {TS_RATIO}, solver={SOLVER}')
    print(f'Feature dimension : {X.shape[1]}')
    print(f'Train/test samples: {n_train} / {n_valid - n_train} (expected)')
    print(f'Delay pairs       : {len(pairs)} (1 <= d1 < d2 <= {MAX_DELAY})')
    print(f'Total train CMC   : {train_scores.sum():.6f}')
    print(f'Total test CMC    : {test_scores.sum():.6f}')
    print(f'Mean test CMC     : {test_scores.mean():.6f}')
    print(f'Max test CMC      : {test_scores[best]:.6f} at {pairs[best]}')
    print(f'Saved to          : {OUTPUT}')
    print('Note: 190 squared correlations can accumulate finite-sample null bias.')


if __name__ == '__main__':
    main()
