# src/run_memory_capacity_validation.py

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.base import TaskData
from src.simulation.runner import simulate_continuous_task

from src.readout.ridge import (
    train_ridge_regression,
    predict_ridge_regression,
)

from src.readout.preprocessing import (
    fit_standardizer,
    transform_standardizer,
)


# ============================================================
# Configuration
# ============================================================

FC = 48e6
TC = 1.0 / FC

TS_RATIOS = [0.025, 0.5]

RESERVOIR_CONFIGS = [
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 2.78),
    (1.0, 8.33),
]

N_SYMBOLS_LIST = [
    3000,
    10000,
    30000,
]

TRAIN_SEED = 42
TEST_SEED = 1234

MAX_DELAY = 200
WASHOUT = 200

K = 8
POINTS_PER_SYMBOL = 16

RIDGE = 1e-6
AMPLITUDE_SCALE = 1e-2

WINDOWS = [
    (1, 10),
    (11, 50),
    (51, 100),
    (101, 200),
]

OUTPUT_DIR = Path("results/memory_capacity_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Result container
# ============================================================

@dataclass
class ValidationResult:
    delays: np.ndarray

    real_mc: np.ndarray
    null_mc: np.ndarray
    excess_mc: np.ndarray

    real_total: float
    null_total: float
    excess_total: float

    window_results: dict

    n_train: int
    n_test: int

    feature_dim: int


# ============================================================
# Input generation
# ============================================================

def generate_input(n_symbols, seed):

    rng = np.random.default_rng(seed)

    return rng.uniform(
        low=-1.0,
        high=1.0,
        size=n_symbols,
    )


# ============================================================
# Reservoir simulation
# ============================================================

def simulate_reservoir(
    eta,
    alpha,
    ts_ratio,
    u,
):

    reservoir = PTSingleReservoir(
        eta=eta,
        alpha=alpha,
    )

    task_data = TaskData(
        name="memory_capacity_validation",
        u=u,
        y=u,
        task_type="regression",
        washout=WASHOUT,
        train_ratio=0.8,
        metadata={
            "eta": eta,
            "alpha": alpha,
            "ts_ratio": ts_ratio,
        },
    )

    solver = (
        "exact_zoh"
        if alpha == 0.0
        else "ivp"
    )

    simulation = simulate_continuous_task(
        reservoir=reservoir,
        task_data=task_data,
        symbol_period=TC * ts_ratio,
        points_per_symbol=POINTS_PER_SYMBOL,
        k=K,
        amplitude_scale=AMPLITUDE_SCALE,
        encoding="zoh",
        carrier_frequency=None,
        solver=solver,
    )

    return np.asarray(simulation.X, dtype=float)


# ============================================================
# Build delayed targets
# ============================================================

def build_delayed_targets(
    u,
    max_delay,
    start,
):

    u = np.asarray(u, dtype=float)

    delays = np.arange(max_delay + 1)

    Y = np.column_stack([
        u[start - d : len(u) - d]
        for d in delays
    ])

    return Y


# ============================================================
# Correlation-based MC
# ============================================================

def squared_correlation(
    y_true,
    y_pred,
):

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if np.std(y_true) < 1e-12:
        return 0.0

    if np.std(y_pred) < 1e-12:
        return 0.0

    corr = np.corrcoef(
        y_true,
        y_pred,
    )[0, 1]

    return float(corr**2)


# ============================================================
# Ridge readout
# ============================================================

def fit_and_predict(
    X_train,
    Y_train,
    X_test,
):

    # Fit normalization using training data only
    mean, std = fit_standardizer(X_train)

    X_train_norm = transform_standardizer(
        X_train,
        mean,
        std,
    )

    X_test_norm = transform_standardizer(
        X_test,
        mean,
        std,
    )

    # Train ridge readout
    W = train_ridge_regression(
        X_train_norm,
        Y_train,
        ridge=RIDGE,
    )

    # Predict independent testing sequence
    Y_pred = predict_ridge_regression(
        X_test_norm,
        W,
    )

    return Y_pred


# ============================================================
# Independent train/test evaluation
# ============================================================

def evaluate_independent_memory(
    X_train,
    u_train,
    X_test,
    u_test,
    null_seed=999,
):

    # Same alignment for all delays
    start = max(
        WASHOUT,
        MAX_DELAY,
    )

    X_train_valid = X_train[start:]
    X_test_valid = X_test[start:]

    Y_train = build_delayed_targets(
        u_train,
        MAX_DELAY,
        start,
    )

    Y_test = build_delayed_targets(
        u_test,
        MAX_DELAY,
        start,
    )

    assert len(X_train_valid) == len(Y_train)
    assert len(X_test_valid) == len(Y_test)

    # --------------------------------------------------------
    # Real readout
    # --------------------------------------------------------

    Y_pred = fit_and_predict(
        X_train_valid,
        Y_train,
        X_test_valid,
    )

    real_mc = np.array([
        squared_correlation(
            Y_test[:, d],
            Y_pred[:, d],
        )
        for d in range(MAX_DELAY + 1)
    ])

    # --------------------------------------------------------
    # Null readout
    #
    # Independent random targets:
    # no relationship with reservoir states
    # --------------------------------------------------------

    N_NULL_PERMUTATIONS = 100

    null_mc_all = []

    for i in range(N_NULL_PERMUTATIONS):

        rng = np.random.default_rng(null_seed + i)

        u_null_train = rng.uniform(
            -1.0,
            1.0,
            size=len(u_train),
        )

        u_null_test = rng.uniform(
            -1.0,
            1.0,
            size=len(u_test),
        )

        Y_null_train = build_delayed_targets(
            u_null_train,
            MAX_DELAY,
            start,
        )

        Y_null_test = build_delayed_targets(
            u_null_test,
            MAX_DELAY,
            start,
        )

        Y_null_pred = fit_and_predict(
            X_train_valid,
            Y_null_train,
            X_test_valid,
        )

        null_mc_i = np.array([
            squared_correlation(
                Y_null_test[:, d],
                Y_null_pred[:, d],
            )
            for d in range(MAX_DELAY + 1)
        ])

        null_mc_all.append(null_mc_i)

    null_mc_all = np.asarray(null_mc_all)

    null_mc = np.mean(
        null_mc_all,
        axis=0,
    )

    null_mc_std = np.std(
        null_mc_all,
        axis=0,
        ddof=1,
    )

    # --------------------------------------------------------
    # Excess MC
    # --------------------------------------------------------

    excess_mc = real_mc - null_mc

    # Delay 0 is excluded from total MC
    real_total = float(
        np.sum(real_mc[1:])
    )

    null_total = float(
        np.sum(null_mc[1:])
    )

    excess_total = float(
        np.sum(excess_mc[1:])
    )

    # --------------------------------------------------------
    # Memory windows
    # --------------------------------------------------------

    window_results = {}

    for d_start, d_end in WINDOWS:

        key = f"{d_start}:{d_end}"

        window_results[key] = {
            "real": float(
                np.sum(real_mc[d_start:d_end + 1])
            ),
            "null": float(
                np.sum(null_mc[d_start:d_end + 1])
            ),
            "excess": float(
                np.sum(excess_mc[d_start:d_end + 1])
            ),
        }

    return ValidationResult(
        delays=np.arange(MAX_DELAY + 1),
        real_mc=real_mc,
        null_mc=null_mc,
        excess_mc=excess_mc,
        real_total=real_total,
        null_total=null_total,
        excess_total=excess_total,
        window_results=window_results,
        n_train=len(X_train_valid),
        n_test=len(X_test_valid),
        feature_dim=X_train.shape[1],
    )


# ============================================================
# Plot MC profile
# ============================================================

def plot_mc_profile(
    result,
    eta,
    alpha,
    ts_ratio,
    n_symbols,
):

    fig, ax = plt.subplots(
        figsize=(9, 5),
    )

    ax.plot(
        result.delays,
        result.real_mc,
        linewidth=2,
        label="Real MC",
    )

    ax.plot(
        result.delays,
        result.null_mc,
        linestyle="--",
        linewidth=2,
        label="Null MC",
    )

    ax.set_xlabel("Delay (symbols)")
    ax.set_ylabel(r"$MC_d$")

    ax.set_title(
        rf"$\eta={eta},\ \alpha={alpha}$"
        "\n"
        rf"$T_s/T_c={ts_ratio},\ N={n_symbols}$"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    filename = (
        f"mc_profile_"
        f"eta{eta}_"
        f"alpha{alpha}_"
        f"ts{ts_ratio}_"
        f"N{n_symbols}.png"
    )

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# Print results
# ============================================================

def print_result(
    result,
    eta,
    alpha,
    ts_ratio,
    n_symbols,
):

    print("\n" + "=" * 70)

    print(
        f"eta={eta}, alpha={alpha}, "
        f"Ts/Tc={ts_ratio}, N={n_symbols}"
    )

    print("=" * 70)

    print(f"Train samples    = {result.n_train}")
    print(f"Test samples     = {result.n_test}")
    print(f"Feature dim      = {result.feature_dim}")

    print(f"MC0              = {result.real_mc[0]:.6f}")
    print(f"MC1              = {result.real_mc[1]:.6f}")

    print(f"Real total MC    = {result.real_total:.6f}")
    print(f"Null total MC    = {result.null_total:.6f}")
    print(f"Excess total MC  = {result.excess_total:.6f}")

    print("\nMemory windows")

    print(
        "Window       Real MC     Null MC     Excess MC"
    )

    print("-" * 55)

    for key, values in result.window_results.items():

        print(
            f"{key:<12}"
            f"{values['real']:>10.6f}"
            f"{values['null']:>12.6f}"
            f"{values['excess']:>14.6f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    summary_rows = []
    window_rows = []

    for ts_ratio in TS_RATIOS:

        for eta, alpha in RESERVOIR_CONFIGS:

            for n_symbols in N_SYMBOLS_LIST:

                print(
                    "\nRunning:",
                    f"Ts/Tc={ts_ratio}",
                    f"eta={eta}",
                    f"alpha={alpha}",
                    f"N={n_symbols}",
                )

                # --------------------------------------------
                # Independent input sequences
                # --------------------------------------------

                u_train = generate_input(
                    n_symbols,
                    TRAIN_SEED,
                )

                u_test = generate_input(
                    n_symbols,
                    TEST_SEED,
                )

                # --------------------------------------------
                # Independent reservoir simulations
                # --------------------------------------------

                X_train = simulate_reservoir(
                    eta,
                    alpha,
                    ts_ratio,
                    u_train,
                )

                X_test = simulate_reservoir(
                    eta,
                    alpha,
                    ts_ratio,
                    u_test,
                )

                # --------------------------------------------
                # Evaluation
                # --------------------------------------------

                result = evaluate_independent_memory(
                    X_train,
                    u_train,
                    X_test,
                    u_test,
                    null_seed=4,
                )

                print_result(
                    result,
                    eta,
                    alpha,
                    ts_ratio,
                    n_symbols,
                )

                plot_mc_profile(
                    result,
                    eta,
                    alpha,
                    ts_ratio,
                    n_symbols,
                )

                # --------------------------------------------
                # Summary
                # --------------------------------------------

                summary_rows.append({
                    "eta": eta,
                    "alpha": alpha,
                    "ts_ratio": ts_ratio,
                    "n_symbols": n_symbols,
                    "n_train": result.n_train,
                    "n_test": result.n_test,
                    "feature_dim": result.feature_dim,
                    "mc0": result.real_mc[0],
                    "mc1": result.real_mc[1],
                    "real_total_mc": result.real_total,
                    "null_total_mc": result.null_total,
                    "excess_total_mc": result.excess_total,
                })

                for key, values in result.window_results.items():

                    window_rows.append({
                        "eta": eta,
                        "alpha": alpha,
                        "ts_ratio": ts_ratio,
                        "n_symbols": n_symbols,
                        "window": key,
                        "real_mc": values["real"],
                        "null_mc": values["null"],
                        "excess_mc": values["excess"],
                    })

    # --------------------------------------------------------
    # Export CSV
    # --------------------------------------------------------

    summary_path = OUTPUT_DIR / "summary.csv"

    with open(
        summary_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    window_path = OUTPUT_DIR / "memory_windows.csv"

    with open(
        window_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=window_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(window_rows)

    print("\nValidation completed.")

    print(f"Summary saved to: {summary_path}")
    print(f"Windows saved to: {window_path}")


if __name__ == "__main__":
    main()