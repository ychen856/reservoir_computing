"""Independent Legendre nonlinear memory capacity (P2/P3) permutation null test.

Copy to src/run_nonlinear_memory_capacity_null.py and run:
    python3 -m src.run_nonlinear_memory_capacity_null

Requires existing src.tasks.properties.nonlinear_memory and _common modules.
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legval

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task
from src.tasks.properties.nonlinear_memory import evaluate_nonlinear_memory
from src.tasks.properties._common import split, score

ETA = 1.0
ALPHA = 8.33
FC = 48e6
TS_RATIO = 0.5
TS = TS_RATIO / FC
INPUT_SCALE = 1e-4
N_SYMBOLS = 3000
SEED = 42
WASHOUT = 200
TRAIN_RATIO = 0.8
POINTS_PER_SYMBOL = 16
K = 8
RIDGE = 1e-6
DELAYS = tuple(range(1, 21))
DEGREES = (2, 3)
SOLVER = 'exact_zoh'
NORMALIZE = True
N_PERMUTATIONS = 100
NULL_SEED = 2026
OUTPUT = Path('results/nonlinear_memory_capacity_null')


def make_targets(u, indices):
    """Match target column ordering in nonlinear_memory.evaluate_nonlinear_memory."""
    cols = []
    for degree in DEGREES:
        coeff = np.zeros(degree + 1)
        coeff[degree] = 1.0
        for delay in DELAYS:
            cols.append(legval(u[indices - delay], coeff))
    return np.column_stack(cols)


def empirical_p(real, null_values):
    """One-sided Monte Carlo p-value, with plus-one correction."""
    return (1 + np.count_nonzero(null_values >= real)) / (len(null_values) + 1)


def main():
    rng = np.random.default_rng(SEED)
    u = rng.uniform(-1.0, 1.0, size=N_SYMBOLS)
    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    task = SimpleNamespace(u=u, y=np.zeros_like(u))
    sim = simulate_continuous_task(
        reservoir=reservoir,
        task_data=task,
        symbol_period=TS,
        points_per_symbol=POINTS_PER_SYMBOL,
        k=K,
        amplitude_scale=INPUT_SCALE,
        encoding='zoh',
        solver=SOLVER,
    )
    X = np.asarray(sim.X, dtype=float)
    if X.shape != (N_SYMBOLS, 2 * K):
        raise RuntimeError(f'Unexpected feature shape: {X.shape}')

    # Use the existing evaluator as the reference implementation.
    real = evaluate_nonlinear_memory(
        X, u,
        delays=DELAYS,
        degrees=DEGREES,
        washout=WASHOUT,
        train_ratio=TRAIN_RATIO,
        ridge=RIDGE,
        normalize=NORMALIZE,
    )
    real_train = np.asarray(real.train_scores, dtype=float)
    real_test = np.asarray(real.test_scores, dtype=float)
    n_targets = len(DEGREES) * len(DELAYS)
    if real_test.size != n_targets:
        raise RuntimeError(f'Expected {n_targets} NMC targets; got {real_test.size}')

    # Identical chronological split and target alignment to nonlinear_memory.py.
    Xa, ua, ia, Xb, ub, ib = split(
        X, u, None, None, max(DELAYS), WASHOUT, TRAIN_RATIO
    )
    Y_train = make_targets(ua, ia)
    Y_test = make_targets(ub, ib)
    check_train, check_test = score(
        Xa, Y_train, Xb, Y_test, RIDGE, NORMALIZE
    )
    if not (np.allclose(check_train, real_train, rtol=1e-8, atol=1e-10)
            and np.allclose(check_test, real_test, rtol=1e-8, atol=1e-10)):
        raise RuntimeError('Real NMC consistency check failed: inspect split/score/target alignment')

    # Null: independently permute whole target rows in training and test.
    # Refit readout for every permutation, preserving within-row target structure.
    null_rng = np.random.default_rng(NULL_SEED)
    null_test = np.empty((N_PERMUTATIONS, n_targets), dtype=float)
    for b in range(N_PERMUTATIONS):
        shuffled_train = Y_train[null_rng.permutation(len(Y_train))]
        shuffled_test = Y_test[null_rng.permutation(len(Y_test))]
        _, null_test[b] = score(
            Xa, shuffled_train, Xb, shuffled_test, RIDGE, NORMALIZE
        )

    null_mean = null_test.mean(axis=0)
    null_p95 = np.quantile(null_test, 0.95, axis=0)
    real_totals = {}
    null_totals = {}
    for j, degree in enumerate(DEGREES):
        sl = slice(j * len(DELAYS), (j + 1) * len(DELAYS))
        real_totals[f'P{degree}'] = float(real_test[sl].sum())
        null_totals[f'P{degree}'] = null_test[:, sl].sum(axis=1)
    real_totals['Total'] = float(real_test.sum())
    null_totals['Total'] = null_test.sum(axis=1)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'nonlinear_memory_null_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['degree', 'delay', 'real_train', 'real_test',
                         'null_test_mean', 'null_test_p95', 'excess_test'])
        for j, degree in enumerate(DEGREES):
            for i, delay in enumerate(DELAYS):
                idx = j * len(DELAYS) + i
                writer.writerow([degree, delay, real_train[idx], real_test[idx],
                                 null_mean[idx], null_p95[idx], real_test[idx] - null_mean[idx]])

    with (OUTPUT / 'null_total_distribution.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['permutation', *[f'P{d}_null_test' for d in DEGREES], 'total_null_test'])
        for b in range(N_PERMUTATIONS):
            writer.writerow([b + 1, *[null_totals[f'P{d}'][b] for d in DEGREES],
                             null_totals['Total'][b]])

    fig, axes = plt.subplots(len(DEGREES), 1, figsize=(9, 7), sharex=True)
    axes = np.atleast_1d(axes)
    for j, (degree, ax) in enumerate(zip(DEGREES, axes)):
        sl = slice(j * len(DELAYS), (j + 1) * len(DELAYS))
        ax.plot(DELAYS, real_test[sl], marker='o', label=f'P{degree} real test')
        ax.plot(DELAYS, null_mean[sl], linestyle='--', label='Null mean')
        ax.fill_between(DELAYS, 0, null_p95[sl], alpha=0.15,
                        label='Per-delay null 95th percentile')
        ax.set_ylabel('Squared correlation')
        ax.set_title(f'P{degree}: real vs. shuffled-target null')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel('Input delay (symbols)')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'nonlinear_memory_null_by_delay.png', dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(DEGREES) + 1, figsize=(14, 4.3))
    for ax, key in zip(axes, [*[f'P{d}' for d in DEGREES], 'Total']):
        values = null_totals[key]
        observed = real_totals[key]
        threshold = float(np.quantile(values, 0.95))
        ax.hist(values, bins=15, edgecolor='black', alpha=0.75)
        ax.axvline(observed, color='tab:red', lw=2, label=f'Real {observed:.4f}')
        ax.axvline(threshold, color='tab:orange', lw=2, ls='--',
                   label=f'Null p95 {threshold:.4f}')
        ax.set_title(f'{key} (p={empirical_p(observed, values):.4f})')
        ax.set_xlabel('Test NMC sum')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'nonlinear_memory_null_distribution.png', dpi=220)
    plt.close(fig)

    print('\n=== Nonlinear Memory Capacity: Null Test (Legendre) ===')
    print(f'Circuit            : eta={ETA}, alpha={ALPHA}')
    print(f'Input              : uniform [-1,1], seed={SEED}, scale={INPUT_SCALE:g} A')
    print(f'Ts/Tc              : {TS_RATIO}, solver={SOLVER}')
    print(f'Feature dimension  : {X.shape[1]}')
    print(f'Train/test samples : {len(Xa)} / {len(Xb)}')
    print(f'Degrees / delays   : {DEGREES} / 1..{max(DELAYS)}')
    print(f'Permutations       : {N_PERMUTATIONS}')
    print('')
    print('Target | Real test | Null mean | Null p95 | Excess | Empirical p')
    print('-' * 72)
    for key in [*[f'P{d}' for d in DEGREES], 'Total']:
        obs = real_totals[key]
        values = null_totals[key]
        print(f'{key:6s} | {obs:9.6f} | {values.mean():9.6f} | '
              f'{np.quantile(values, .95):8.6f} | {obs - values.mean():+8.6f} | '
              f'{empirical_p(obs, values):.4f}')
    print(f'Saved to           : {OUTPUT}')
    print('Note: per-delay thresholds are uncorrected for multiple comparisons.')
    print('Note: permutation significance is conditional on this input realization.')


if __name__ == '__main__':
    main()
