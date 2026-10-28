"""Independent PT nonlinear memory capacity (Legendre P2/P3) experiment.

Place at src/run_nonlinear_memory_capacity.py and run:
    python3 -m src.run_nonlinear_memory_capacity

Requires existing src.tasks.properties.nonlinear_memory and _common modules.
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task
from src.tasks.properties.nonlinear_memory import evaluate_nonlinear_memory

# Fixed physical circuit and sampling settings (no task-dependent sweep)
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
OUTPUT = Path('results/nonlinear_memory_capacity')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # Legendre targets require nondegenerate continuous input on [-1,1].
    # Bipolar +/-1 makes P2 constant and P3 identical to linear input.
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
    X = np.asarray(sim.X)
    if X.shape != (N_SYMBOLS, 2 * K):
        raise RuntimeError(f'Unexpected feature shape {X.shape}')

    result = evaluate_nonlinear_memory(
        X, u,
        delays=DELAYS,
        degrees=DEGREES,
        washout=WASHOUT,
        train_ratio=TRAIN_RATIO,
        ridge=RIDGE,
        normalize=True,
    )
    tr = np.asarray(result.train_scores, dtype=float)
    te = np.asarray(result.test_scores, dtype=float)
    if len(tr) != len(DEGREES) * len(DELAYS) or len(te) != len(tr):
        raise RuntimeError('Unexpected NMC target count')

    rows = []
    for j, degree in enumerate(DEGREES):
        for i, delay in enumerate(DELAYS):
            idx = j * len(DELAYS) + i
            rows.append(dict(degree=degree, delay=delay, target=result.labels[idx],
                             train_nmc=float(tr[idx]), test_nmc=float(te[idx])))
    with (OUTPUT / 'nonlinear_memory_scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, degree in enumerate(DEGREES):
        sl = slice(j * len(DELAYS), (j + 1) * len(DELAYS))
        ax.plot(DELAYS, te[sl], marker='o', label=f'P{degree} test')
        ax.plot(DELAYS, tr[sl], linestyle='--', alpha=0.65, label=f'P{degree} train')
    ax.set(xlabel='Input delay (symbols)', ylabel='Squared correlation',
           title=f'PT Single: Nonlinear Memory Capacity ($\\eta$={ETA}, $\\alpha$={ALPHA})')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'nonlinear_memory_capacity.png', dpi=220)
    plt.close(fig)

    print('\n=== Nonlinear Memory Capacity (Legendre) ===')
    print(f'Circuit          : eta={ETA}, alpha={ALPHA}')
    print(f'Input            : uniform [-1,1], seed={SEED}, scale={INPUT_SCALE} A')
    print(f'Ts/Tc            : {TS_RATIO}, solver={SOLVER}')
    print(f'Feature dimension: {X.shape[1]}')
    print(f'Symbols / washout: {N_SYMBOLS} / {WASHOUT}')
    print(f'Degrees / delays : {DEGREES} / 1..{max(DELAYS)}')
    for j, degree in enumerate(DEGREES):
        sl = slice(j * len(DELAYS), (j + 1) * len(DELAYS))
        print(f'P{degree} train/test NMC: {tr[sl].sum():.6f} / {te[sl].sum():.6f}')
    print(f'Total train NMC  : {tr.sum():.6f}')
    print(f'Total test NMC   : {te.sum():.6f}')
    print(f'Saved to         : {OUTPUT}')
    print('Note: P2 is even under input inversion; P3 is odd. This is not a null-corrected result.')


if __name__ == '__main__':
    main()
