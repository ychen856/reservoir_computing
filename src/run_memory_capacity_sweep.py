import numpy as np
import matplotlib.pyplot as plt

from src.tasks.base import TaskData

from src.circuits.pt_single import (
    PTSingleReservoir,
)

from src.simulation.experiment import (
    PropertyExperiment,
    PropertyExperimentConfig,
)

from src.tasks.properties.memory_capacity import evaluate_linear_memory_capacity

def build_input_history(u, max_delay):

    u = np.asarray(u)

    X = np.zeros(
        (len(u), max_delay + 1)
    )

    for d in range(max_delay + 1):

        X[d:, d] = u[:len(u) - d]

    return X

def main():

    # ========================================================
    # Parameters
    # ========================================================

    FC = 48e6
    TC = 1.0 / FC

    N_CYCLES_PER_SYMBOL = 1 / 40
    T_SYMBOL = N_CYCLES_PER_SYMBOL * TC

    N_SYMBOLS = 3000

    WASHOUT = 200
    TRAIN_RATIO = 0.8

    K = 8
    MAX_DELAY = 200

    RIDGE = 1e-6

    # ========================================================
    # Generate random input
    # ========================================================

    rng = np.random.default_rng(42)

    u = rng.uniform(
        low=-1.0,
        high=1.0,
        size=N_SYMBOLS,
    )

    task_data = TaskData(
        name="linear_memory_capacity",
        u=u,
        y=u,
        task_type="regression",
        washout=WASHOUT,
        train_ratio=TRAIN_RATIO,
        metadata={
            "input_distribution": "uniform(-1,1)",
            "seed": 42,
        },
    )

    # ========================================================
    # Reservoir
    # ========================================================

    reservoir = PTSingleReservoir(
        eta=1.0,
        #alpha=0.0,
        #alpha=2.78,
        alpha=8.33,
    )

    # ========================================================
    # Property experiment
    # ========================================================

    config = PropertyExperimentConfig(
        symbol_period=T_SYMBOL,
        points_per_symbol=16,
        k=K,
        amplitude_scale=1e-2,
        encoding="zoh",

        property_type="linear_memory_capacity",

        max_delay=MAX_DELAY,
        ridge=RIDGE,
        normalize=True,
    )

    experiment = PropertyExperiment(
        reservoir=reservoir,
        task_data=task_data,
        config=config,
        reservoir_name="PT Single",
    )

    # ========================================================
    # Simulation
    # ========================================================

    simulation = experiment.simulate()

    # ========================================================
    # Evaluation
    # ========================================================

    result = experiment.evaluate(
        simulation=simulation,
    )

    ev = result.evaluation

    print("\n=== Linear Memory Capacity ===")

    print(f"Feature dimension: {ev.feature_dim}")
    print(f"Train samples    : {ev.n_train}")
    print(f"Test samples     : {ev.n_test}")

    print(
        f"Total train MC   : "
        f"{ev.total_train_capacity:.6f}"
    )

    print(
        f"Total test MC    : "
        f"{ev.total_test_capacity:.6f}"
    )

    print("\nDelay | Train MC | Test MC")
    print("--------------------------")

    for d, mc_train, mc_test in zip(
        ev.delays,
        ev.train_capacity,
        ev.test_capacity,
    ):

        print(
            f"{d:5d} | "
            f"{mc_train:.6f} | "
            f"{mc_test:.6f}"
        )

    # ========================================================
    # Plot
    # ========================================================

    '''plt.figure(figsize=(7, 4))

    plt.plot(
        ev.delays,
        ev.train_capacity,
        "o-",
        label="Train",
    )

    plt.plot(
        ev.delays,
        ev.test_capacity,
        "s-",
        label="Test",
    )

    plt.xlabel("Input delay (symbols)")
    plt.ylabel("Linear memory capacity")

    plt.title("PT Single: Linear Memory Capacity")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()'''

    TS_RATIOS = [
        1/80,
        1/40,
        1/20,
        1/10,
        1/5,
        1/2,
        1,
    ]

    results = []

    for ratio in TS_RATIOS:

        T_SYMBOL = ratio * TC

        config = PropertyExperimentConfig(
            symbol_period=T_SYMBOL,
            points_per_symbol=16,
            k=8,
            amplitude_scale=1e-2,
            encoding="zoh",
            property_type="linear_memory_capacity",
            max_delay=MAX_DELAY,
            ridge=1e-6,
            normalize=True,
        )

        experiment = PropertyExperiment(
            reservoir=reservoir,
            task_data=task_data,
            config=config,
            reservoir_name="PT Single",
        )

        result = experiment.run()
        ev = result.evaluation

        mc0 = ev.test_capacity[ev.delays == 0][0]
        mc1 = ev.test_capacity[ev.delays == 1][0]

        results.append({
            "Ts/Tc": ratio,
            "MC0": mc0,
            "MC1": mc1,
            "Total_MC": ev.total_test_capacity,
        })

        print(
            f"Ts/Tc={ratio:.5f} | "
            f"MC0={mc0:.6f} | "
            f"MC1={mc1:.6f} | "
            f"Total MC={ev.total_test_capacity:.6f}"
        )


if __name__ == "__main__":
    main()