# src/run_memory_capacity_null.py

import numpy as np
import matplotlib.pyplot as plt

from src.circuits.pt_single import PTSingleReservoir

from src.tasks.base import TaskData

from src.simulation.experiment import (
    PropertyExperiment,
    PropertyExperimentConfig,
)

from src.tasks.properties.memory_capacity_null import (
    evaluate_memory_capacity_null,
)


# ==================================================
# Configuration
# ==================================================

FC = 48e6
TC = 1.0 / FC

N_SYMBOLS = 3000

WASHOUT = 200
TRAIN_RATIO = 0.8

POINTS_PER_SYMBOL = 16
K = 8

I_SCALE = 1e-4

RIDGE = 1e-6

MAX_DELAY = 200
N_PERMUTATIONS = 30

SEED = 42

# Start with the two most informative symbol periods.
TS_RATIOS = [
    #1 / 40,
    1 / 2,
]

PARAMETER_CASES = [
    #(0.0, 0.0),
    #(1.0, 0.0),
    #(1.0, 2.78),
    (1.0, 8.33),
]


# ==================================================
# Input data
# ==================================================

rng = np.random.default_rng(SEED)

'''u = rng.uniform(
    -1.0,
    1.0,
    size=N_SYMBOLS,
)'''
u = rng.uniform(-1.0, 1.0, size=N_SYMBOLS)

task_data = TaskData(
    name="linear_memory_capacity_null",
    u=u,
    y=u,
    task_type="regression",
    washout=WASHOUT,
    train_ratio=TRAIN_RATIO,
)


# ==================================================
# Simulation and evaluation
# ==================================================

results = {}

for ts_ratio in TS_RATIOS:

    T_SYMBOL = TC * ts_ratio

    for eta, alpha in PARAMETER_CASES:

        print()
        print("=" * 65)
        print(
            f"eta={eta}, alpha={alpha}, "
            f"Ts/Tc={ts_ratio:.5f}"
        )
        print("=" * 65)

        reservoir = PTSingleReservoir(
            eta=eta,
            alpha=alpha,
        )

        config = PropertyExperimentConfig(
            symbol_period=T_SYMBOL,
            points_per_symbol=POINTS_PER_SYMBOL,
            k=K,
            amplitude_scale=I_SCALE,
            encoding="zoh",
            property_type="linear_memory_capacity",
            max_delay=MAX_DELAY,
            ridge=RIDGE,
            normalize=True,
            solver='exact_zoh'
        )

        experiment = PropertyExperiment(
            reservoir=reservoir,
            task_data=task_data,
            config=config,
        )

        # Only one ODE simulation per configuration.
        simulation = experiment.simulate()

        result = evaluate_memory_capacity_null(
            X=simulation.X,
            u=task_data.u,
            max_delay=MAX_DELAY,
            washout=WASHOUT,
            train_ratio=TRAIN_RATIO,
            ridge=RIDGE,
            normalize=True,
            n_permutations=N_PERMUTATIONS,
            seed=SEED,
        )

        results[(ts_ratio, eta, alpha)] = result

        print(
            f"Real total MC   = "
            f"{result.real_total_mc:.6f}"
        )

        print(
            f"Null total MC   = "
            f"{result.null_total_mc:.6f}"
        )

        print(
            f"Excess total MC = "
            f"{result.excess_total_mc:.6f}"
        )

        print()
        print(
            "Delay | Real MC | Null MC | Excess MC"
        )
        print("-" * 48)

        selected_delays = [1, 2, 5, 10, 20, 50, 100, 150, 200]

        for delay in selected_delays:
            idx = np.where(result.delays == delay)[0]

            if len(idx) == 0:
                continue

            i = idx[0]

            print(
                f"{delay:5d} | "
                f"{result.real_test[i]:.6f} | "
                f"{result.null_test_mean[i]:.6f} | "
                f"{result.excess_test[i]:+.6f}"
            )


# ==================================================
# Plot
# ==================================================

for ts_ratio in TS_RATIOS:

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
    )

    for ax, (eta, alpha) in zip(
        axes.flat,
        PARAMETER_CASES,
    ):

        result = results[
            (ts_ratio, eta, alpha)
        ]

        d = result.delays

        ax.plot(
            d,
            result.real_test,
            linewidth=2,
            label="Real MC",
        )

        ax.plot(
            d,
            result.null_test_mean,
            "--",
            linewidth=2,
            label="Null MC",
        )

        ax.fill_between(
            d,
            np.maximum(
                0,
                result.null_test_mean
                - result.null_test_std,
            ),
            result.null_test_mean
            + result.null_test_std,
            alpha=0.2,
        )

        ax.set_title(
            rf"$\eta={eta},\ \alpha={alpha}$"
        )

        ax.set_xlabel("Delay (symbols)")
        ax.set_ylabel(r"$MC_d$")

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

    fig.suptitle(
        rf"Linear Memory Capacity: "
        rf"$T_s/T_c={ts_ratio:.5f}$",
        fontsize=15,
    )

    plt.tight_layout()
    plt.show()