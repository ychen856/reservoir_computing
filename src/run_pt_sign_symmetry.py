"""Check global input-sign symmetry of the fixed PT reservoir.

Copy to src/run_pt_sign_symmetry.py; run:
    python3 -m src.run_pt_sign_symmetry

Uses exactly the same input encoding, circuit, and temporal features as the
CMC runner. Both simulations start from the same zero initial state.
"""
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task

FC = 48e6
TS_RATIO = 0.5
ETA, ALPHA = 1.0, 8.33
INPUT_SCALE = 1e-4
N_SYMBOLS = 3000
SEED = 42
POINTS_PER_SYMBOL = 16
K = 8
OUTPUT = Path('results/pt_sign_symmetry')


def get_features(u):
    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    task = SimpleNamespace(u=u, y=np.zeros_like(u))
    sim = simulate_continuous_task(
        reservoir=reservoir,
        task_data=task,
        symbol_period=TS_RATIO / FC,
        points_per_symbol=POINTS_PER_SYMBOL,
        k=K,
        amplitude_scale=INPUT_SCALE,
        encoding='zoh',
        solver='exact_zoh',
    )
    X = np.asarray(sim.X, dtype=float)
    if X.shape != (N_SYMBOLS, 2 * K):
        raise ValueError(f'Unexpected feature shape: {X.shape}')
    if not np.all(np.isfinite(X)):
        raise ValueError('Nonfinite reservoir features')
    return X


def main():
    rng = np.random.default_rng(SEED)
    u = rng.choice([-1.0, 1.0], size=N_SYMBOLS)
    X_pos = get_features(u)
    X_neg = get_features(-u)

    residual = X_pos + X_neg
    denom = np.linalg.norm(X_pos) + np.linalg.norm(X_neg)
    epsilon = np.linalg.norm(residual) / denom if denom else np.nan
    peak = max(np.max(np.abs(X_pos)), np.max(np.abs(X_neg)))
    relative_peak = np.max(np.abs(residual)) / peak if peak else np.nan

    print('\n=== PT input-sign symmetry check ===')
    print(f'eta={ETA}, alpha={ALPHA}, input_scale={INPUT_SCALE:g} A')
    print(f'Ts/Tc={TS_RATIO}, N={N_SYMBOLS}, K={K}, seed={SEED}')
    print(f'Feature shape            : {X_pos.shape}')
    print(f'||X(u)+X(-u)|| / (||X(u)||+||X(-u)||): {epsilon:.9e}')
    print(f'Max |X(u)+X(-u)|         : {np.max(np.abs(residual)):.9e} V')
    print(f'Max residual / max signal: {relative_peak:.9e}')
    print('Interpretation: near-zero residual supports numerical odd symmetry.')
    print('No universal pass threshold: compare with solver convergence/tolerances.')

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / 'sign_symmetry_features.npz',
                        u=u, X_pos=X_pos, X_neg=X_neg, residual=residual)

    # Compare a representative temporal node from each tank for the first 200 symbols.
    n_plot = min(200, N_SYMBOLS)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, col, name in zip(axes, [0, K], ['V1', 'V2']):
        ax.plot(X_pos[:n_plot, col], label='X(u)', lw=1)
        ax.plot(-X_neg[:n_plot, col], '--', label='-X(-u)', lw=1)
        ax.set_ylabel(f'{name} [V]')
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel('Symbol index (first temporal node)')
    fig.suptitle(f'PT sign inversion: normalized residual = {epsilon:.3e}')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'sign_symmetry.png', dpi=180)
    plt.close(fig)
    print(f'Saved to: {OUTPUT}')


if __name__ == '__main__':
    main()
