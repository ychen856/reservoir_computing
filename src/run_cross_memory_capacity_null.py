"""Null test for PT single-tank cross memory capacity.

Copy to src/run_cross_memory_capacity_null.py and run:
    python3 -m src.run_cross_memory_capacity_null

Uses the same circuit/input protocol as run_cross_memory_capacity.py and
reuses the existing cross_memory / _common readout implementation.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task
from src.tasks.properties.cross_memory import evaluate_cross_memory
from src.tasks.properties._common import split, score

FC = 48e6
TS_RATIO = 0.5
SYMBOL_PERIOD = TS_RATIO / FC
ETA, ALPHA = 1.0, 8.33
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
N_PERMUTATIONS = 100
NULL_SEED = 2026
ENCODING = 'zoh'
SOLVER = 'exact_zoh'
OUTPUT = Path('results/cross_memory_capacity_null')


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
    X = np.asarray(sim.X, dtype=float)
    if X.shape != (N_SYMBOLS, 2 * K):
        raise ValueError(f'Unexpected feature shape: {X.shape}')

    pairs = [(i, j) for i in range(1, MAX_DELAY + 1)
             for j in range(i + 1, MAX_DELAY + 1)]

    # Real scores: EXACTLY the existing CMC evaluation.
    real = evaluate_cross_memory(
        X=X, u=u, pairs=pairs, max_delay=MAX_DELAY,
        washout=WASHOUT, train_ratio=TRAIN_RATIO,
        ridge=RIDGE, normalize=NORMALIZE,
    )
    real_train = np.asarray(real.train_scores, dtype=float)
    real_test = np.asarray(real.test_scores, dtype=float)

    # Reuse the same split and ridge/scoring functions as cross_memory.py.
    Xa, ua, ia, Xb, ub, ib = split(
        X, u, None, None, max(j for _, j in pairs), WASHOUT, TRAIN_RATIO
    )
    Y_train = np.column_stack([ua[ia - i] * ua[ia - j] for i, j in pairs])
    Y_test = np.column_stack([ub[ib - i] * ub[ib - j] for i, j in pairs])
    check_train, check_test = score(
        Xa, Y_train, Xb, Y_test, RIDGE, NORMALIZE
    )
    if not (np.allclose(check_train, real_train, rtol=1e-8, atol=1e-10)
            and np.allclose(check_test, real_test, rtol=1e-8, atol=1e-10)):
        raise RuntimeError('Real-score consistency check failed; inspect _common.split/score')

    # Null hypothesis: no relationship between X and the second-order targets.
    # Shuffle whole target ROWS independently in train and test. This preserves
    # target marginal distributions and within-row pair relationships.
    # Retrain the readout for EACH permutation.
    null_rng = np.random.default_rng(NULL_SEED)
    null_test = np.empty((N_PERMUTATIONS, len(pairs)), dtype=float)
    for b in range(N_PERMUTATIONS):
        Y_train_null = Y_train[null_rng.permutation(len(Y_train))]
        Y_test_null = Y_test[null_rng.permutation(len(Y_test))]
        _, null_test[b] = score(
            Xa, Y_train_null, Xb, Y_test_null, RIDGE, NORMALIZE
        )

    null_mean = null_test.mean(axis=0)
    null_p95 = np.quantile(null_test, 0.95, axis=0)
    real_total = float(real_test.sum())
    null_totals = null_test.sum(axis=1)
    null_total_mean = float(null_totals.mean())
    null_total_p95 = float(np.quantile(null_totals, 0.95))
    # Monte Carlo one-sided p-value, with plus-one correction.
    p_total = (1 + np.count_nonzero(null_totals >= real_total)) / (N_PERMUTATIONS + 1)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'cross_memory_null_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['delay_1', 'delay_2', 'real_train', 'real_test',
                         'null_test_mean', 'null_test_p95', 'excess_test'])
        for (i, j), tr, te, nm, p95 in zip(
                pairs, real_train, real_test, null_mean, null_p95):
            writer.writerow([i, j, tr, te, nm, p95, te - nm])

    with (OUTPUT / 'null_total_distribution.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['permutation', 'null_total_test_cmc'])
        for b, total in enumerate(null_totals, start=1):
            writer.writerow([b, total])

    # Heatmap: real minus null mean, rather than uncorrected CMC.
    excess_map = np.full((MAX_DELAY, MAX_DELAY), np.nan)
    for (i, j), excess in zip(pairs, real_test - null_mean):
        excess_map[j - 1, i - 1] = excess
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = np.nanmax(np.abs(excess_map))
    im = ax.imshow(excess_map, origin='lower', interpolation='nearest',
                   extent=(0.5, MAX_DELAY + 0.5, 0.5, MAX_DELAY + 0.5),
                   vmin=-vmax, vmax=vmax, cmap='coolwarm', aspect='equal')
    ax.set(xlabel='First delay $d_1$ (symbols)',
           ylabel='Second delay $d_2$ (symbols)',
           title=rf'CMC above null mean ($\eta$={ETA}, $\alpha$={ALPHA})')
    fig.colorbar(im, ax=ax, label='Real test CMC - null mean')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'cross_memory_excess_heatmap.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null_totals, bins=20, edgecolor='black', alpha=0.75,
            label='Permuted-target null')
    ax.axvline(real_total, color='tab:red', lw=2, label=f'Real = {real_total:.4f}')
    ax.axvline(null_total_p95, color='tab:orange', ls='--', lw=2,
               label=f'Null 95th pct = {null_total_p95:.4f}')
    ax.set(xlabel='Total test CMC (190 pairs)', ylabel='Count',
           title='CMC null distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'cross_memory_null_total.png', dpi=200)
    plt.close(fig)

    print('\n=== Cross Memory Capacity: Null Test ===')
    print(f'Circuit            : eta={ETA}, alpha={ALPHA}')
    print(f'Input              : bipolar +/-1, seed={SEED}, scale={INPUT_SCALE:g} A')
    print(f'Ts/Tc              : {TS_RATIO}, solver={SOLVER}')
    print(f'Feature dimension  : {X.shape[1]}')
    print(f'Train/test samples : {len(Xa)} / {len(Xb)}')
    print(f'Delay pairs        : {len(pairs)}')
    print(f'Permutations       : {N_PERMUTATIONS}')
    print(f'Real total test CMC: {real_total:.6f}')
    print(f'Null total mean    : {null_total_mean:.6f}')
    print(f'Null total 95th pct: {null_total_p95:.6f}')
    print(f'Excess over mean   : {real_total - null_total_mean:+.6f}')
    print(f'Empirical p (total): {p_total:.4f} (one-sided, Monte Carlo)')
    print(f'Pairs > null p95   : {int(np.sum(real_test > null_p95))} / {len(pairs)} (uncorrected)')
    print(f'Saved to           : {OUTPUT}')
    print('Note: per-pair 95th percentiles are not multiple-comparison corrected.')


if __name__ == '__main__':
    main()
