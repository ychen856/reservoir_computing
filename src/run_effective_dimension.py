"""Standalone effective-dimension experiment for the fixed PT reservoir.

Place at src/run_effective_dimension.py and run:
    python3 -m src.run_effective_dimension
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task
from src.tasks.properties.effective_dimension import evaluate_effective_dimension

FC = 48e6
TS_RATIO = 0.5
TS = TS_RATIO / FC
ETA = 1.0
ALPHA = 8.33
INPUT_SCALE = 1e-4
SEED = 42
N_SYMBOLS = 1600
WASHOUT = 200
POINTS_PER_SYMBOL = 16
K = 8
SOLVER = "exact_zoh"
OUTPUT = Path("results/effective_dimension")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    u = rng.uniform(-1.0, 1.0, N_SYMBOLS)
    task = SimpleNamespace(u=u, y=np.zeros(N_SYMBOLS, dtype=float))
    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    simulation = simulate_continuous_task(
        reservoir=reservoir,
        task_data=task,
        symbol_period=TS,
        points_per_symbol=POINTS_PER_SYMBOL,
        k=K,
        amplitude_scale=INPUT_SCALE,
        encoding="zoh",
        solver=SOLVER,
    )
    X = np.asarray(simulation.X, dtype=float)
    if X.shape != (N_SYMBOLS, 2 * K):
        raise RuntimeError(f"Unexpected feature shape: {X.shape}")
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite reservoir features")

    X_used = X[WASHOUT:]
    # Match the original combined runner: centered covariance,
    # no per-feature standardization.
    result = evaluate_effective_dimension(X_used, standardize=False)
    eig = np.asarray(result.eigenvalues, dtype=float)
    fractions = eig / eig.sum() if eig.sum() > 0 else np.zeros_like(eig)
    cumulative = np.cumsum(fractions)

    with (OUTPUT / "eigenvalue_spectrum.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "covariance_eigenvalue",
                         "variance_fraction", "cumulative_variance"])
        for i, (lam, frac, cum) in enumerate(zip(eig, fractions, cumulative), 1):
            writer.writerow([i, float(lam), float(frac), float(cum)])

    np.savez(
        OUTPUT / "features.npz",
        X=X_used, eigenvalues=eig, variance_fractions=fractions,
        eta=ETA, alpha=ALPHA, input_scale=INPUT_SCALE,
        ts_ratio=TS_RATIO, seed=SEED, washout=WASHOUT,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    indices = np.arange(1, len(eig) + 1)
    ax.semilogy(indices, np.maximum(fractions, np.finfo(float).tiny),
                "o-", label="Variance fraction")
    ax.set(
        xlabel="Principal-component index",
        ylabel="Explained variance fraction (log scale)",
        title=f"PT Single: Effective Dimension = {result.dimension:.3f}",
    )
    ax.set_xticks(indices)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "effective_dimension_spectrum.png", dpi=200)
    plt.close(fig)

    print("\n=== Effective Dimension ===")
    print(f"Circuit           : eta={ETA}, alpha={ALPHA}")
    print(f"Input             : uniform [-1,1], seed={SEED}, scale={INPUT_SCALE:g} A")
    print(f"Ts/Tc             : {TS_RATIO}, solver={SOLVER}")
    print(f"Symbols / washout : {N_SYMBOLS} / {WASHOUT}")
    print(f"Feature matrix    : {X_used.shape}")
    print(f"Standardize       : False (mean-center only)")
    print(f"Effective dimension: {result.dimension:.6f} / {result.feature_dim}")
    print("Variance fractions:", np.array2string(fractions, precision=6))
    print(f"Saved to          : {OUTPUT}")


if __name__ == "__main__":
    main()
