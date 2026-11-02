"""Standalone mackey_glass benchmark: python3 -m src.run_mackey_glass.

No dependency on remaining_benchmarks or _remaining_runner.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from src.circuits.pt_single import PTSingleReservoir
from src.tasks.benchmarks.mackey_glass import generate_mackey_glass
from src.simulation.experiment import ExperimentConfig, ReadoutExperiment
from src.simulation.evaluation import (
    evaluate_continuous_regression,
    evaluate_independent_classification,
)

# ============================================================
# Configuration (edit here; no implicit solver selection)
# ============================================================
N_SAMPLES = 3000
WASHOUT = 200
TRAIN_RATIO = 0.8
SEED = 42
HORIZONS = [20]
FC = 48e6
TS_RATIO = 1.0
POINTS_PER_SYMBOL = 16
K = 8
I_SCALE = 1e-4
RIDGE = 1e-6
ETA = 1.1
ALPHA = 8.33
SOLVER = "exact_zoh"
HISTORY_TAPS = (1, 10, 16)


def history_features(u, taps):
    """Build [u[n], ..., u[n-taps+1]] without circular wraparound."""
    u = np.asarray(u, dtype=float).reshape(-1)
    if not 1 <= taps <= len(u):
        raise ValueError("Invalid history length")
    X = np.zeros((len(u), taps), dtype=float)
    for lag in range(taps):
        X[lag:, lag] = u[:len(u)-lag]
    return X


def evaluate_baseline(X, task):
    """Same chronological split and washout as ReadoutExperiment."""
    X = np.asarray(X)[task.washout:]
    y = np.asarray(task.y)[task.washout:]
    if task.task_type == "classification":
        split = int(len(y) * task.train_ratio)
        return evaluate_independent_classification(
            X_train=X[:split], y_train=y[:split],
            X_test=X[split:], y_test=y[split:],
            ridge=RIDGE, normalize=True, n_classes=2,
        )
    return evaluate_continuous_regression(
        X=X, y=y, washout=0, train_ratio=task.train_ratio,
        ridge=RIDGE, normalize=True,
    )


def print_result(label, ev):
    print(f"\n=== {label} ===")
    print(f"Feature dim : {ev.feature_dim}")
    print(f"Train/Test NRMSE : {ev.train_nrmse:.6f} / {ev.test_nrmse:.6f}")
    print(f"Train/Test r     : {ev.train_r_per_output[0]:.6f} / {ev.test_r_per_output[0]:.6f}")


def run_one(value):
    task = generate_mackey_glass(
        n_samples=N_SAMPLES, washout=WASHOUT,
        train_ratio=TRAIN_RATIO, horizon=value,
    )
    if task.washout < max(HISTORY_TAPS) - 1:
        raise ValueError("Washout must cover all input-history taps")


    # 16-tap input history
    X_history_16 = history_features(task.u, 16)

    # Quadratic expansion
    poly = PolynomialFeatures(
        degree=2,
        include_bias=False
    )

    X_history_16_quad = poly.fit_transform(X_history_16)

    print("Quadratic history feature dimension:", X_history_16_quad.shape[1])

    result_quad = evaluate_continuous_regression(
        X_history_16_quad,
        task.y,
        washout=task.washout,
        train_ratio=task.train_ratio,
        ridge=RIDGE,
        normalize=True,
    )

    print("\n=== 16-tap quadratic input history ===")
    print("Feature dim :", result_quad.feature_dim)

    print(
        "Train/Test NRMSE : "
        f"{result_quad.train_nrmse:.6f} / "
        f"{result_quad.test_nrmse:.6f}"
    )

    print(
        "Train/Test r     : "
        f"{result_quad.train_r_per_output[0]:.6f} / "
        f"{result_quad.test_r_per_output[0]:.6f}"
    )


    n_valid = len(task.u) - task.washout
    n_train = int(n_valid * task.train_ratio)
    print(f"\nTask: {task.name}; metadata: {task.metadata}")
    print(f"Valid samples={n_valid}, train={n_train}, test={n_valid-n_train}")
    print(f"Ts/Tc={TS_RATIO}, K={K}, I_scale={I_SCALE:g}, ridge={RIDGE:g}")

    print_result("Current input", evaluate_baseline(task.u[:, None], task))
    for taps in HISTORY_TAPS:
        if taps == 1:
            continue
        print_result(f"{taps}-tap history", evaluate_baseline(history_features(task.u, taps), task))

    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    config = ExperimentConfig(
        symbol_period=TS_RATIO / FC,
        points_per_symbol=POINTS_PER_SYMBOL,
        k=K,
        amplitude_scale=I_SCALE,
        encoding="zoh",
        ridge=RIDGE,
        normalize=True,
        solver=SOLVER,
        evaluation_type="regression",
    )
    experiment = ReadoutExperiment(
        reservoir=reservoir, task_data=task,
        config=config, reservoir_name="PT Single",
    )
    simulation = experiment.simulate()
    result = experiment.evaluate(simulation=simulation, k=K, ridge=RIDGE)
    print_result("PT Single reservoir", result.evaluation)
    

    return result


if __name__ == "__main__":
    for value in HORIZONS:
        run_one(value)
