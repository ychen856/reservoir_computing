"""Standalone parity benchmark: python3 -m src.run_parity.

No dependency on remaining_benchmarks or _remaining_runner.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix
from src.circuits.pt_single import PTSingleReservoir
from src.tasks.benchmarks.parity import generate_parity
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
WINDOWS = (2, 3, 5, 10)
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

def parity_cubic_sanity_check(task, ridge=1e-6):
    """
    Cubic input-history sanity check for Parity-3.

    Assumptions:
        task.u: shape (N,), bipolar input {-1, +1}
        task.y: shape (N,), parity labels {0, 1}
        task.washout: number of initial samples excluded
        task.train_ratio: training fraction
    """

    u = np.asarray(task.u).reshape(-1)
    y = np.asarray(task.y).reshape(-1).astype(int)

    window = 3
    N = len(u)

    # --------------------------------------------------
    # Construct causal input-history features
    # h[n] = [u[n], u[n-1], u[n-2]]
    # --------------------------------------------------

    X_history = np.zeros((N, window), dtype=float)

    for delay in range(window):
        X_history[delay:, delay] = u[:N - delay]

    # --------------------------------------------------
    # Cubic polynomial expansion
    # --------------------------------------------------

    poly = PolynomialFeatures(
        degree=3,
        include_bias=False
    )

    X_poly = poly.fit_transform(X_history)

    # --------------------------------------------------
    # Apply the same washout and chronological split
    # --------------------------------------------------

    start = max(task.washout, window - 1)

    X_valid = X_poly[start:]
    y_valid = y[start:]

    n_train = int(len(y_valid) * task.train_ratio)

    X_train = X_valid[:n_train]
    X_test = X_valid[n_train:]

    y_train = y_valid[:n_train]
    y_test = y_valid[n_train:]

    # --------------------------------------------------
    # Ridge regression + binary classification
    # --------------------------------------------------

    model = Ridge(
        alpha=ridge,
        fit_intercept=True
    )

    model.fit(X_train, y_train)

    train_scores = model.predict(X_train)
    test_scores = model.predict(X_test)

    train_pred = (train_scores >= 0.5).astype(int)
    test_pred = (test_scores >= 0.5).astype(int)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\n=== Parity-3 cubic history sanity check ===")
    print(f"Feature dimension : {X_poly.shape[1]}")
    print(f"Train accuracy    : {train_acc:.6f}")
    print(f"Test accuracy     : {test_acc:.6f}")
    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    return train_acc, test_acc

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


def run_one(value):
    task = generate_parity(
        n_samples=N_SAMPLES, washout=WASHOUT,
        train_ratio=TRAIN_RATIO, window=value,
        seed=SEED,
    )
    if task.washout < max(HISTORY_TAPS) - 1:
        raise ValueError("Washout must cover all input-history taps")

    if value == 3:
        parity_cubic_sanity_check(
            task,
            ridge=RIDGE,
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
    return result


if __name__ == "__main__":
    for value in WINDOWS:
        run_one(value)
