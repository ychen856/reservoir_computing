"""Standalone PT Single generalization-rank (GR) experiment.

Run from project root:
    python3 -m src.run_generalization_rank
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task
from src.tasks.properties.generalization_rank import evaluate_generalization_rank

FC = 48e6
TS_RATIO = 0.5
TS = TS_RATIO / FC
ETA = 1.0
ALPHA = 8.33
INPUT_SCALE = 1e-4
SEED = 44  # Same trial RNG stream as standalone KR.
N_TRIALS = 24
TRIAL_SYMBOLS = 48
PERTURBATION_STD = 0.02
POINTS_PER_SYMBOL = 16
K = 8
RANK_TOLERANCE = 1e-3
SOLVER = "exact_zoh"
OUTPUT = Path("results/generalization_rank")


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

    # Reproduce original trial_matrices() random-number stream:
    # base -> independent history -> perturbation, repeated for each trial.
    base = rng.uniform(-1.0, 1.0, size=TRIAL_SYMBOLS)
    X = []
    for _ in range(N_TRIALS):
        _ = rng.uniform(-1.0, 1.0, size=TRIAL_SYMBOLS)  # KR RNG draw
        perturbed = np.clip(
            base + rng.normal(0.0, PERTURBATION_STD, size=TRIAL_SYMBOLS),
            -1.0, 1.0,
        )
        X.append(get_final_features(reservoir, perturbed))
    X = np.asarray(X)

    result = evaluate_generalization_rank(
        X, relative_tolerance=RANK_TOLERANCE
    )
    singular_values = np.asarray(result.singular_values)
    normalized = (
        singular_values / singular_values[0]
        if singular_values.size and singular_values[0] > 0
        else np.zeros_like(singular_values)
    )

    with (OUTPUT / "singular_values.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "component", "singular_value", "normalized_singular_value",
            "above_rank_threshold"
        ])
        for i, (sv, sn) in enumerate(zip(singular_values, normalized), 1):
            writer.writerow([
                i, float(sv), float(sn), bool(sn > RANK_TOLERANCE)
            ])

    np.savez(
        OUTPUT / "trial_features.npz",
        X=X, base_input=base, singular_values=singular_values,
        eta=ETA, alpha=ALPHA, input_scale=INPUT_SCALE,
        ts_ratio=TS_RATIO, seed=SEED, perturbation_std=PERTURBATION_STD,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(
        np.arange(1, len(normalized) + 1),
        np.maximum(normalized, np.finfo(float).tiny),
        "o-", label="Normalized singular values"
    )
    ax.axhline(
        RANK_TOLERANCE, color="tab:red", linestyle="--",
        label=f"Rank threshold = {RANK_TOLERANCE:g}"
    )
    ax.set(
        xlabel="Singular-value index",
        ylabel="Normalized singular value",
        title=f"PT Single: Generalization Rank (GR = {result.rank})",
    )
    ax.set_xticks(np.arange(1, len(normalized) + 1))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "generalization_rank_spectrum.png", dpi=200)
    plt.close(fig)

    print("\n=== Generalization Rank ===")
    print(f"Circuit          : eta={ETA}, alpha={ALPHA}")
    print(f"Input            : base uniform [-1,1], seed={SEED}, scale={INPUT_SCALE:g} A")
    print(f"Perturbation std : {PERTURBATION_STD}")
    print(f"Ts/Tc            : {TS_RATIO}, solver={SOLVER}")
    print(f"Trials / symbols : {N_TRIALS} / {TRIAL_SYMBOLS}")
    print(f"Feature matrix   : {X.shape}")
    print(f"Rank tolerance   : {RANK_TOLERANCE:g}")
    print(f"Generalization rank: {result.rank} / {result.feature_dim}")
    print("Normalized singular values:",
          np.array2string(normalized, precision=5))
    print(f"Saved to         : {OUTPUT}")


if __name__ == "__main__":
    main()
