"""Standalone PT Single kernel-rank experiment.

Run from project root:
    python3 -m src.run_kernel_rank
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task
from src.tasks.properties.kernel_rank import evaluate_kernel_rank

FC = 48e6
TS_RATIO = 0.5
TS = TS_RATIO / FC
ETA = 1.0
ALPHA = 8.33
INPUT_SCALE = 1e-4
SEED = 44  # Original combined runner used SEED + 2 (42 + 2).
N_TRIALS = 24
TRIAL_SYMBOLS = 48
POINTS_PER_SYMBOL = 16
K = 8
RANK_TOLERANCE = 1e-3
SOLVER = "exact_zoh"
OUTPUT = Path("results/kernel_rank")


def get_final_features(reservoir, u):
    task = SimpleNamespace(u=np.asarray(u, dtype=float),
                           y=np.zeros(len(u), dtype=float))
    result = simulate_continuous_task(
        reservoir=reservoir,
        task_data=task,
        symbol_period=TS,
        points_per_symbol=POINTS_PER_SYMBOL,
        k=K,
        amplitude_scale=INPUT_SCALE,
        encoding="zoh",
        solver=SOLVER,
    )
    if result.X.shape != (len(u), 2 * K):
        raise RuntimeError(f"Unexpected feature shape: {result.X.shape}")
    if not np.isfinite(result.X).all():
        raise RuntimeError("Non-finite reservoir features")
    return result.X[-1].copy()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)

    # Keep the original combined-runner RNG consumption: it drew a base
    # history and a GR perturbation in every trial. The latter is unused
    # here but retained so the KR histories match the old trial_matrices().
    base = rng.uniform(-1.0, 1.0, size=TRIAL_SYMBOLS)
    X = []
    for _ in range(N_TRIALS):
        u = rng.uniform(-1.0, 1.0, size=TRIAL_SYMBOLS)
        _ = np.clip(base + rng.normal(0.0, 0.02, size=TRIAL_SYMBOLS),
                    -1.0, 1.0)
        X.append(get_final_features(reservoir, u))
    X = np.asarray(X)

    result = evaluate_kernel_rank(X, relative_tolerance=RANK_TOLERANCE)
    s = result.singular_values
    normalized = s / s[0] if len(s) and s[0] > 0 else np.zeros_like(s)
    threshold = RANK_TOLERANCE
    # Secondary diagnostics: do not replace the original rank definition.
    energy = s**2 / np.sum(s**2) if np.sum(s**2) > 0 else np.zeros_like(s)

    with (OUTPUT / "singular_values.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "singular_value",
                         "normalized_singular_value", "energy_fraction",
                         "above_rank_threshold"])
        for i, (sv, sn, e) in enumerate(zip(s, normalized, energy), 1):
            writer.writerow([i, float(sv), float(sn), float(e),
                             bool(sn > threshold)])

    np.savez(OUTPUT / "trial_features.npz", X=X, singular_values=s,
             eta=ETA, alpha=ALPHA, input_scale=INPUT_SCALE,
             ts_ratio=TS_RATIO, seed=SEED)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.arange(1, len(s) + 1),
                np.maximum(normalized, np.finfo(float).tiny), "o-",
                label="Normalized singular values")
    ax.axhline(threshold, color="tab:red", linestyle="--",
               label=f"Rank threshold = {threshold:g}")
    ax.set(xlabel="Singular-value index", ylabel="Normalized singular value",
           title=f"PT Single: Kernel Rank (KR = {result.rank})")
    ax.set_xticks(np.arange(1, len(s) + 1))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "kernel_rank_spectrum.png", dpi=200)
    plt.close(fig)

    print("\n=== Kernel Rank ===")
    print(f"Circuit          : eta={ETA}, alpha={ALPHA}")
    print(f"Input            : uniform [-1,1], seed={SEED}, scale={INPUT_SCALE:g} A")
    print(f"Ts/Tc            : {TS_RATIO}, solver={SOLVER}")
    print(f"Trials / symbols : {N_TRIALS} / {TRIAL_SYMBOLS}")
    print(f"Feature matrix   : {X.shape}")
    print(f"Rank tolerance   : {RANK_TOLERANCE:g} (relative to largest singular value)")
    print(f"Kernel rank      : {result.rank} / {result.feature_dim}")
    print("Normalized singular values:", np.array2string(normalized, precision=5))
    print(f"Saved to         : {OUTPUT}")


if __name__ == "__main__":
    main()
