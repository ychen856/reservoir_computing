import numpy as np

from dataclasses import dataclass

from src.readout.ridge import (
    train_ridge_regression,
    predict_ridge_regression,
)

from src.readout.preprocessing import (
    fit_standardizer,
    transform_standardizer,
)



# ============================================================
# Linear Memory Capacity Result
# ============================================================

@dataclass
class LinearMemoryCapacityResult:

    delays: np.ndarray

    train_capacity: np.ndarray
    test_capacity: np.ndarray

    total_train_capacity: float
    total_test_capacity: float

    n_train: int
    n_test: int

    feature_dim: int
    ridge: float


# ============================================================
# Utility
# ============================================================

def squared_correlation(y_true, y_pred):
    """
    Squared Pearson correlation.

    Returns 0 if either signal has zero variance.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if np.std(y_true) < 1e-12:
        return 0.0

    if np.std(y_pred) < 1e-12:
        return 0.0

    r = np.corrcoef(y_true, y_pred)[0, 1]

    return float(r * r)


# ============================================================
# Linear Memory Capacity
# ============================================================

def evaluate_linear_memory_capacity(
    X,
    u,
    max_delay=20,
    washout=100,
    train_ratio=0.8,
    ridge=1e-6,
    normalize=True,
):
    """
    Evaluate linear memory capacity.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Reservoir feature matrix.

    u : ndarray, shape (N,)
        Original input sequence.

    max_delay : int
        Maximum input delay.

    washout : int
        Number of initial samples discarded.

    train_ratio : float
        Chronological train/test split.

    ridge : float
        Ridge regularization.

    normalize : bool
        Whether to standardize reservoir features.

    Returns
    -------
    LinearMemoryCapacityResult
    """

    X = np.asarray(X, dtype=float)
    u = np.asarray(u, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if len(X) != len(u):
        raise ValueError("X and u must have the same length")

    if max_delay < 1:
        raise ValueError("max_delay must be >= 1")

    # --------------------------------------------------------
    # Build delayed targets
    # --------------------------------------------------------

    delays = np.arange(1, max_delay + 1)

    # Use a common valid interval for every delay.
    # This ensures MC_d values are directly comparable.

    start = max(washout, max_delay)

    X_valid = X[start:]

    Y_valid = np.column_stack([
        u[start - d:len(u) - d]
        for d in delays
    ])

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    n_valid = len(X_valid)
    n_train = int(n_valid * train_ratio)

    X_train = X_valid[:n_train]
    X_test = X_valid[n_train:]

    Y_train = Y_valid[:n_train]
    Y_test = Y_valid[n_train:]

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    if normalize:

        mu, std = fit_standardizer(X_train)

        X_train = transform_standardizer(
            X_train, mu, std
        )

        X_test = transform_standardizer(
            X_test, mu, std
        )

    # --------------------------------------------------------
    # Train linear readout
    # --------------------------------------------------------

    W = train_ridge_regression(
        X_train,
        Y_train,
        ridge=ridge,
    )

    Y_train_pred = predict_ridge_regression(
        X_train,
        W,
    )

    Y_test_pred = predict_ridge_regression(
        X_test,
        W,
    )

    # --------------------------------------------------------
    # Memory capacity for each delay
    # --------------------------------------------------------

    train_capacity = np.array([
        squared_correlation(
            Y_train[:, i],
            Y_train_pred[:, i],
        )
        for i in range(len(delays))
    ])

    test_capacity = np.array([
        squared_correlation(
            Y_test[:, i],
            Y_test_pred[:, i],
        )
        for i in range(len(delays))
    ])

    return LinearMemoryCapacityResult(
        delays=delays,

        train_capacity=train_capacity,
        test_capacity=test_capacity,

        total_train_capacity=float(
            np.sum(train_capacity)
        ),

        total_test_capacity=float(
            np.sum(test_capacity)
        ),

        n_train=len(X_train),
        n_test=len(X_test),

        feature_dim=X.shape[1],
        ridge=ridge,
    )