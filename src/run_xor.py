"""Standalone xor benchmark: python3 -m src.run_xor.

No dependency on remaining_benchmarks or _remaining_runner.
"""
import numpy as np
from dataclasses import replace
from src.circuits.pt_single import PTSingleReservoir
from src.tasks.benchmarks.xor import generate_xor
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
DELAYS = (1, 2, 5, 10)
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
RUN_SYMMETRY_TEST = True  # Extra reservoir simulation for each delay
RUN_QUADRATIC_READOUT = True


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
    print(f"Train/Test accuracy : {ev.train_accuracy:.6f} / {ev.test_accuracy:.6f}")
    print("Test confusion matrix:\n", ev.confusion_matrix)


def xor_polynomial_features(u, delay):
    """Exact XOR interaction: [u[n], u[n-delay], u[n]*u[n-delay]]."""
    u = np.asarray(u, dtype=float).reshape(-1)
    X = np.zeros((len(u), 3), dtype=float)
    X[:, 0] = u
    X[delay:, 1] = u[:-delay]
    X[delay:, 2] = u[delay:] * u[:-delay]
    return X


def quadratic_features(X):
    """Linear + all degree-2 monomials, including squares."""
    X = np.asarray(X, dtype=float)
    i, j = np.triu_indices(X.shape[1])
    return np.column_stack((X, X[:, i] * X[:, j]))


def symmetry_diagnostic(X_pos, X_neg, washout):
    """Global input-sign inversion symmetry (not pairwise XOR invariance)."""
    A = np.asarray(X_pos)[washout:]
    B = np.asarray(X_neg)[washout:]
    if A.shape != B.shape:
        raise ValueError(f"Symmetry feature shape mismatch: {A.shape} vs {B.shape}")
    denom = np.linalg.norm(A) + np.linalg.norm(B)
    err = np.linalg.norm(A + B) / denom if denom > 0 else float('nan')
    print(f"Global sign symmetry error = {err:.6e} (0 means X(-u)=-X(u))")
    return err


def run_one(value):
    task = generate_xor(
        n_samples=N_SAMPLES, washout=WASHOUT,
        train_ratio=TRAIN_RATIO, delay=value,
        seed=SEED,
    )
    print("Input unique values:", np.unique(task.u))
    print("Input mean:", np.mean(task.u))
    print("Input min/max:", np.min(task.u), np.max(task.u))

    if task.washout < max(HISTORY_TAPS) - 1:
        raise ValueError("Washout must cover all input-history taps")

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

    print_result(
        f"Exact XOR polynomial history (delay={value})",
        evaluate_baseline(xor_polynomial_features(task.u, value), task),
    )

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
        evaluation_type="sequence_classification",
        n_classes=2,
    )
    experiment = ReadoutExperiment(
        reservoir=reservoir, task_data=task,
        config=config, reservoir_name="PT Single",
    )
    simulation = experiment.simulate()
    result = experiment.evaluate(simulation=simulation, k=K, ridge=RIDGE)
    print_result("PT Single reservoir", result.evaluation)


    X = simulation.X

    print("\nReservoir feature statistics:")
    print("X shape:", X.shape)
    print("X min/max:", X.min(), X.max())
    print("X mean:", X.mean())
    print("X std:", X.std())
    print("First feature mean:", X[:, 0].mean())
    print("First feature std:", X[:, 0].std())


    if RUN_QUADRATIC_READOUT:
        # Diagnostic only: higher-dimensional nonlinear readout, not the
        # standard linear-readout reservoir benchmark.
        X_quad = quadratic_features(simulation.X)
        print_result(
            "PT Single + quadratic readout (diagnostic)",
            evaluate_baseline(X_quad, task),
        )

    if RUN_SYMMETRY_TEST:
        # Re-run from zero initial state with every input sign inverted.
        # Use a fresh reservoir instance to avoid any possible state reuse.
        inverted_task = replace(task, u=-np.asarray(task.u).copy())
        inverted_experiment = ReadoutExperiment(
            reservoir=PTSingleReservoir(eta=ETA, alpha=ALPHA),
            task_data=inverted_task,
            config=config,
            reservoir_name="PT Single (inverted input)",
        )
        inverted_simulation = inverted_experiment.simulate()
        symmetry_diagnostic(simulation.X, inverted_simulation.X, task.washout)

    return result


if __name__ == "__main__":
    for value in DELAYS:
        run_one(value)
