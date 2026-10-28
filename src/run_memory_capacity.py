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

    N_CYCLES_PER_SYMBOL = 1 / 2
    T_SYMBOL = N_CYCLES_PER_SYMBOL * TC

    N_SYMBOLS = 3000

    WASHOUT = 200
    TRAIN_RATIO = 0.8

    K = 8
    MAX_DELAY = 20

    RIDGE = 1e-6

    # ========================================================
    # Generate random input
    # ========================================================

    rng = np.random.default_rng(42)

    '''u = rng.uniform(
        low=-1.0,
        high=1.0,
        size=N_SYMBOLS,
    )'''
    u = rng.choice(
        [-1.0, 1.0],
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
            "input_distribution": "bipolar(-1,+1)",
            "seed": 42,
        },
    )

    # ========================================================
    # Reservoir
    # ========================================================

    reservoir = PTSingleReservoir(
        eta=1.0,
        alpha=8.33,
    )

    # ========================================================
    # Property experiment
    # ========================================================

    config = PropertyExperimentConfig(
        symbol_period=T_SYMBOL,
        points_per_symbol=16,
        k=K,
        amplitude_scale=1e-4,
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

    plt.figure(figsize=(7, 4))

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
    plt.show()



    # ============================================================
    # Input History Baseline
    # ============================================================

    print("\n=== Input History Baseline ===")

    # Build delayed input features
    X_history = build_input_history(
        u=task_data.u,
        max_delay=MAX_DELAY,
    )

    print(f"Input history shape: {X_history.shape}")

    # Evaluate memory capacity
    baseline = evaluate_linear_memory_capacity(
        X=X_history,
        u=task_data.u,
        max_delay=MAX_DELAY,
        washout=WASHOUT,
        train_ratio=TRAIN_RATIO,
        ridge=RIDGE,
        normalize=True,
    )

    print(f"Feature dimension: {baseline.feature_dim}")
    print(f"Train samples    : {baseline.n_train}")
    print(f"Test samples     : {baseline.n_test}")

    print(
        f"Total train MC   : "
        f"{baseline.total_train_capacity:.6f}"
    )

    print(
        f"Total test MC    : "
        f"{baseline.total_test_capacity:.6f}"
    )

    print("\nDelay | Train MC | Test MC")
    print("--------------------------")

    for d, mc_train, mc_test in zip(
        baseline.delays,
        baseline.train_capacity,
        baseline.test_capacity,
    ):

        print(
            f"{d:5d} | "
            f"{mc_train:.6f} | "
            f"{mc_test:.6f}"
        )

if __name__ == "__main__":
    main()