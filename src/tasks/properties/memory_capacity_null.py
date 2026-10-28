# src/tasks/properties/memory_capacity_null.py

from dataclasses import dataclass

import numpy as np

from src.tasks.properties.memory_capacity import (
    evaluate_linear_memory_capacity,
)


@dataclass
class MemoryCapacityNullResult:
    delays: np.ndarray

    real_train: np.ndarray
    real_test: np.ndarray

    null_train_mean: np.ndarray
    null_train_std: np.ndarray

    null_test_mean: np.ndarray
    null_test_std: np.ndarray

    excess_test: np.ndarray

    real_total_mc: float
    null_total_mc: float
    excess_total_mc: float

    n_permutations: int


def evaluate_memory_capacity_null(
    X,
    u,
    max_delay=200,
    washout=200,
    train_ratio=0.8,
    ridge=1e-6,
    normalize=True,
    n_permutations=30,
    seed=42,
):
    """
    Compare real linear memory capacity against shuffled-target null.

    Real:
        X[k] -> u[k-d]

    Null:
        X[k] -> shuffled u[k-d]

    Train and test targets are independently permuted.

    Total MC excludes delay 0.
    """

    X = np.asarray(X, dtype=float)
    u = np.asarray(u, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must have shape (N, feature_dim)")

    if len(X) != len(u):
        raise ValueError("X and u must have the same length")

    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")

    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")

    # --------------------------------------------------
    # 1. Real MC
    # --------------------------------------------------

    real = evaluate_linear_memory_capacity(
        X=X,
        u=u,
        max_delay=max_delay,
        washout=washout,
        train_ratio=train_ratio,
        ridge=ridge,
        normalize=normalize,
    )

    delays = real.delays

    # --------------------------------------------------
    # 2. Prepare identical valid samples
    # --------------------------------------------------

    start = max(washout, max_delay)

    X_valid = X[start:]

    Y_valid = np.column_stack([
        u[start - d:len(u) - d]
        for d in delays
    ])

    n_valid = len(X_valid)
    n_train = int(n_valid * train_ratio)

    if n_train < 2 or n_valid - n_train < 2:
        raise ValueError("Not enough train/test samples")

    X_train = X_valid[:n_train]
    X_test = X_valid[n_train:]

    Y_train = Y_valid[:n_train]
    Y_test = Y_valid[n_train:]

    # --------------------------------------------------
    # 3. Train-only feature standardization
    # --------------------------------------------------

    if normalize:
        x_mean = X_train.mean(axis=0)
        x_std = X_train.std(axis=0)

        x_std = np.where(x_std < 1e-12, 1.0, x_std)

        X_train = (X_train - x_mean) / x_std
        X_test = (X_test - x_mean) / x_std

    # --------------------------------------------------
    # 4. Ridge regression preparation
    # --------------------------------------------------

    # Bias is not regularized.
    X_train_aug = np.column_stack([
        np.ones(len(X_train)),
        X_train,
    ])

    X_test_aug = np.column_stack([
        np.ones(len(X_test)),
        X_test,
    ])

    gram = X_train_aug.T @ X_train_aug

    reg = ridge * np.eye(gram.shape[0])
    reg[0, 0] = 0.0

    A = gram + reg

    # Reuse factorization structure for every permutation.
    # Pseudoinverse is robust to rank-deficient features.
    A_pinv = np.linalg.pinv(A)

    projection = A_pinv @ X_train_aug.T

    # --------------------------------------------------
    # 5. Null permutations
    # --------------------------------------------------

    rng = np.random.default_rng(seed)

    n_delays = len(delays)

    null_train = np.zeros(
        (n_permutations, n_delays),
        dtype=float,
    )

    null_test = np.zeros(
        (n_permutations, n_delays),
        dtype=float,
    )

    def corr_squared_columns(y_true, y_pred):
        """
        Column-wise squared Pearson correlation.
        """
        yt = y_true - y_true.mean(axis=0)
        yp = y_pred - y_pred.mean(axis=0)

        numerator = np.sum(yt * yp, axis=0)

        denominator = np.sqrt(
            np.sum(yt**2, axis=0)
            * np.sum(yp**2, axis=0)
        )

        corr = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )

        return corr**2

    for p in range(n_permutations):

        # Independently shuffle each delay target.
        Y_train_null = np.empty_like(Y_train)
        Y_test_null = np.empty_like(Y_test)

        for j in range(n_delays):
            Y_train_null[:, j] = rng.permutation(
                Y_train[:, j]
            )

            Y_test_null[:, j] = rng.permutation(
                Y_test[:, j]
            )

        # Train W on shuffled training targets.
        W_null = projection @ Y_train_null

        pred_train = X_train_aug @ W_null
        pred_test = X_test_aug @ W_null

        null_train[p] = corr_squared_columns(
            Y_train_null,
            pred_train,
        )

        null_test[p] = corr_squared_columns(
            Y_test_null,
            pred_test,
        )

    # --------------------------------------------------
    # 6. Summary
    # --------------------------------------------------

    null_train_mean = null_train.mean(axis=0)
    null_train_std = null_train.std(axis=0)

    null_test_mean = null_test.mean(axis=0)
    null_test_std = null_test.std(axis=0)

    excess_test = (
        real.test_capacity - null_test_mean
    )

    # Exclude delay 0.
    real_total = float(np.sum(real.test_capacity))
    null_total = float(np.sum(null_test_mean))

    # Signed excess: do not clip each delay before summing.
    excess_total = real_total - null_total

    return MemoryCapacityNullResult(
        delays=delays,

        real_train=real.train_capacity,
        real_test=real.test_capacity,

        null_train_mean=null_train_mean,
        null_train_std=null_train_std,

        null_test_mean=null_test_mean,
        null_test_std=null_test_std,

        excess_test=excess_test,

        real_total_mc=real_total,
        null_total_mc=null_total,
        excess_total_mc=excess_total,

        n_permutations=n_permutations,
    )