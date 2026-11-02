from src.circuits.pt_single import PTSingleReservoir

from src.tasks.benchmarks.sine_cube import (
    generate_sine_cube,
)

from src.simulation.experiment import (
    ExperimentConfig,
    ReadoutExperiment,
)


def main():

    # ========================================================
    # Parameters
    # ========================================================

    FC = 48e6
    TC = 1.0 / FC

    N_CYCLES_PER_SYMBOL = 1 / 40

    T_SYMBOL = N_CYCLES_PER_SYMBOL * TC

    N_SAMPLES = 2000

    K = 8
    RIDGE = 1e-6

    # ========================================================
    # Task
    # ========================================================

    task_data = generate_sine_cube(
        n_samples=N_SAMPLES,
        washout=100,
        train_ratio=0.8,
    )

    # ========================================================
    # Reservoir
    # ========================================================

    reservoir = PTSingleReservoir(
        eta=1.0,
        alpha=8.33,
    )

    # ========================================================
    # Experiment
    # ========================================================

    config = ExperimentConfig(
        symbol_period=T_SYMBOL,
        points_per_symbol=16,
        k=K,
        amplitude_scale=1e-2,
        encoding="zoh",
        ridge=RIDGE,
        normalize=True,
        evaluation_type="regression",
    )

    experiment = ReadoutExperiment(
        reservoir=reservoir,
        task_data=task_data,
        config=config,
        reservoir_name="PT Single",
    )

    #result = experiment.run()
    simulation = experiment.simulate()

    result = experiment.evaluate(
        simulation=simulation,
    )

    # ========================================================
    # Results
    # ========================================================

    print("\n=== Experiment Interface Test ===")

    print(f"Task          : {result.task_name}")
    print(f"Reservoir     : {result.reservoir_name}")

    print(
        f"Feature dim   : "
        f"{result.evaluation.feature_dim}"
    )

    print(
        f"Train samples : "
        f"{result.evaluation.n_train}"
    )

    print(
        f"Test samples  : "
        f"{result.evaluation.n_test}"
    )

    print(
        f"Train NRMSE   : "
        f"{result.evaluation.train_nrmse:.6f}"
    )

    print(
        f"Test NRMSE    : "
        f"{result.evaluation.test_nrmse:.6f}"
    )


if __name__ == "__main__":
    main()