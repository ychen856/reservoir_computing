"""
Nonlinear PAM4 Distortion Strength Sweep

Run:
    python3 -m src.run_pam4_nonlinear_distortion_sweep

Experiments:
    1. Input-only baseline
    2. 5-tap linear equalizer
    3. PT single-symbol
    4. PT full temporal window (L=16, 20)

Channel:
    z = q + s * (0.036*q^2 - 0.011*q^3)

Each distortion strength uses:
    - Same PAM4 symbol sequence
    - Same train/test indices
    - One independent reservoir simulation

Output:
    pam4_nonlinear_distortion_sweep.csv
"""

from pathlib import Path
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
from itertools import combinations_with_replacement

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.applications.pam4_nonlinear import (
    generate_pam4_nonlinear_task,
)
from src.simulation.runner import simulate_continuous_task


# ============================================================
# 1. Configuration
# ============================================================

N_SYMBOLS = 3000
WASHOUT = 200
TRAIN_RATIO = 0.8
SEED = 42

CHANNEL_RADIUS = 2

# Reservoir timing
FC = 48e6

N_CYCLES_PER_SYMBOL = 1 / 2

T_SYMBOL = N_CYCLES_PER_SYMBOL / FC

POINTS_PER_SYMBOL = 16
K = 8

# Reservoir parameters
PULSE_DUTY_VALUE =1.0
ENERGY_MATCHED = False
BASE_I_SCALE = 3e-4
I_SCALE = (
    BASE_I_SCALE / np.sqrt(PULSE_DUTY_VALUE)
    if ENERGY_MATCHED
    else BASE_I_SCALE
)
ALPHA = 8.33

# Readout
RIDGE = 1e-1 #1e-6

# Nonlinear channel
BASE_QUAD = 0.036
BASE_CUBIC = -0.011

DISTORTION_STRENGTHS = [2.5] #(
    #1,
    #1.5,
    #2,
    #2.5,
    #3,
    #4,
#)


# Noise
ADD_AWGN = False
SNR_DB = 40.0
NOISE_SEED = 123

# ------------------------------------------------------------
# Experiment control
# ------------------------------------------------------------

# False:
#   Only evaluate input-only and 5-tap equalizer.
#
# True:
#   Also run PT reservoir simulations.

RUN_RESERVOIR = True

# PT temporal windows
PT_WINDOWS = [20] #(0, 16, 20)

# Output
OUTPUT_CSV = Path(
    "pam4_nonlinear_distortion_sweep.csv"
)


# ============================================================
# 2. Temporal Window
# ============================================================

def build_future_window(X, L):
    """
    Construct future temporal window.

    X_window[n] = [
        X[n],
        X[n+1],
        ...
        X[n+L]
    ]

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Reservoir features.

    L : int
        Future observation window length.

    Returns
    -------
    ndarray, shape (N, D*(L+1))
    """

    X = np.asarray(X, dtype=float)

    print("X min:", np.min(X))
    print("X max:", np.max(X))
    print("X std:", np.std(X))

    N, D = X.shape

    X_window = np.full(
        (N, D * (L + 1)),
        np.nan,
        dtype=float,
    )

    for offset in range(L + 1):

        start = offset * D
        stop = (offset + 1) * D

        X_window[:N-L, start:stop] = (
            X[offset:N-L+offset]
        )

    return X_window


# ============================================================
# 3. Scalar Ridge Regression
# ============================================================
def polynomial_features(X, degree=2):
    """
    Generate polynomial features without bias.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Original input features.

    degree : int
        Maximum polynomial degree.

    Returns
    -------
    X_poly : ndarray
        Polynomial feature matrix.

    Examples
    --------
    D = 5

    degree=1:
        5 features

    degree=2:
        20 features

    degree=3:
        55 features

    Notes
    -----
    The bias term is excluded because evaluate()
    already handles the regression intercept.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D feature matrix."
        )

    if degree < 1:
        raise ValueError(
            "degree must be >= 1."
        )

    N, D = X.shape

    features = []

    for current_degree in range(1, degree + 1):

        combinations = combinations_with_replacement(
            range(D),
            current_degree,
        )

        for indices in combinations:

            feature = np.prod(
                X[:, indices],
                axis=1,
            )

            features.append(feature)

    X_poly = np.column_stack(features)

    return X_poly



def evaluate(
    X,
    y,
    train_idx,
    test_idx,
    ridge=RIDGE,
):
    """
    Scalar ridge regression for PAM4 classification.

    Steps:
        1. Train-only normalization
        2. Ridge regression
        3. Continuous prediction
        4. PAM4 quantization
        5. Accuracy / SER / NRMSE
    """

    X_train = np.asarray(
        X[train_idx],
        dtype=float,
    )

    X_test = np.asarray(
        X[test_idx],
        dtype=float,
    )

    y_train = np.asarray(
        y[train_idx],
        dtype=float,
    )

    y_test = np.asarray(
        y[test_idx],
        dtype=float,
    )

    # --------------------------------------------------------
    # Check features
    # --------------------------------------------------------

    if not (
        np.isfinite(X_train).all()
        and np.isfinite(X_test).all()
    ):
        raise ValueError(
            "Nonfinite features detected. "
            "Check temporal window boundaries."
        )

    # --------------------------------------------------------
    # Train-only normalization
    # --------------------------------------------------------

    mean = X_train.mean(axis=0)

    std = X_train.std(axis=0)

    std[std < 1e-12] = 1.0

    X_train = (X_train - mean) / std

    X_test = (X_test - mean) / std

    # --------------------------------------------------------
    # Ridge regression
    # --------------------------------------------------------

    y_mean = y_train.mean()

    D = X_train.shape[1]

    A = (
        X_train.T @ X_train
        + ridge * np.eye(D)
    )

    b = X_train.T @ (
        y_train - y_mean
    )

    W = np.linalg.solve(A, b)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    pred_train = (
        X_train @ W + y_mean
    )

    pred_test = (
        X_test @ W + y_mean
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def score(target, pred):

        labels = np.clip(
            np.rint(pred),
            0,
            3,
        ).astype(int)

        accuracy = float(
            np.mean(labels == target)
        )

        rmse = np.sqrt(
            np.mean((pred - target) ** 2)
        )

        nrmse = float(
            rmse / np.std(target)
        )

        return accuracy, nrmse

    train_acc, train_nrmse = score(
        y_train,
        pred_train,
    )

    test_acc, test_nrmse = score(
        y_test,
        pred_test,
    )

    return {
        "feature_dim": D,

        "n_train": len(train_idx),
        "n_test": len(test_idx),

        "train_accuracy": train_acc,
        "test_accuracy": test_acc,

        "train_nrmse": train_nrmse,
        "test_nrmse": test_nrmse,

        "train_ser": 1 - train_acc,
        "test_ser": 1 - test_acc,
    }




def evaluate_multiclass(
    X,
    y,
    train_idx,
    test_idx,
    ridge=1e-1,
    n_classes=4,
):
    """
    Multiclass ridge classification.

    X:          (N, D) feature matrix
    y:          (N,) integer class labels
    train_idx:  training indices
    test_idx:   testing indices
    ridge:      regularization strength
    n_classes:  number of classes

    Returns:
        Dictionary containing accuracy, SER, and readout statistics.
    """

    X_train = np.asarray(X[train_idx], dtype=np.float64)
    X_test = np.asarray(X[test_idx], dtype=np.float64)

    y_train = np.asarray(y[train_idx], dtype=np.int64)
    y_test = np.asarray(y[test_idx], dtype=np.int64)


    # ------------------------------------
    # 1. Feature standardization
    # ------------------------------------
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std = np.where(std < 1e-12, 1.0, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # ------------------------------------
    # 2. One-hot encoding
    # ------------------------------------
    Y_train = np.eye(n_classes)[y_train]

    # ------------------------------------
    # 3. Center target
    # ------------------------------------
    y_mean = Y_train.mean(axis=0)
    Y_centered = Y_train - y_mean

    # ------------------------------------
    # 4. Ridge regression
    # ------------------------------------
    # SVD formulation avoids forming X.T @ X,
    # improving numerical stability.
    U, s, Vt = np.linalg.svd(
        X_train,
        full_matrices=False,
    )

    if ridge == 0:
        cutoff = np.finfo(float).eps * max(X_train.shape) * s[0]
        factors = np.divide(
            1.0,
            s,
            out=np.zeros_like(s),
            where=s > cutoff,
        )
    else:
        factors = s / (s**2 + ridge)

    W = (Vt.T * factors) @ (U.T @ Y_centered)

    # ------------------------------------
    # 5. Class scores
    # ------------------------------------
    train_scores = X_train @ W + y_mean
    test_scores = X_test @ W + y_mean

    train_pred = np.argmax(train_scores, axis=1)
    test_pred = np.argmax(test_scores, axis=1)

    # ------------------------------------
    # 6. Metrics
    # ------------------------------------
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)

    
    print("True distribution:")
    print(np.bincount(y_test, minlength=n_classes))
    
    print("Predicted distribution:")
    print(np.bincount(test_pred, minlength=n_classes))

    print(confusion_matrix(y_test, test_pred))

    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "train_ser": float(1.0 - train_acc),
        "test_ser": float(1.0 - test_acc),
        "weight_norm": float(np.linalg.norm(W)),
        "weight_max": float(np.max(np.abs(W))),
        "feature_dim": int(X.shape[1]),
        "W": W,
    }



def evaluate_ordinal(
    X,
    y,
    train_idx,
    test_idx,
    ridge=1e-1,
    n_classes=4,
):
    """
    Ordinal Ridge Classification.

    For PAM4:
        Class 0 -> [0, 0, 0]
        Class 1 -> [1, 0, 0]
        Class 2 -> [1, 1, 0]
        Class 3 -> [1, 1, 1]

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Feature matrix.

    y : ndarray, shape (N,)
        Integer class labels.

    train_idx, test_idx : ndarray
        Training and testing indices.

    ridge : float
        Ridge regularization strength.

    n_classes : int
        Number of ordered classes.

    Returns
    -------
    dict
        Accuracy, SER, weight statistics, and predictions.
    """

    X_train = np.asarray(X[train_idx], dtype=np.float64)
    X_test = np.asarray(X[test_idx], dtype=np.float64)

    y_train = np.asarray(y[train_idx], dtype=np.int64)
    y_test = np.asarray(y[test_idx], dtype=np.int64)

    # --------------------------------------------------
    # 1. Feature standardization
    # --------------------------------------------------
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std = np.where(std < 1e-12, 1.0, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # --------------------------------------------------
    # 2. Ordinal targets
    # --------------------------------------------------
    thresholds = np.arange(n_classes - 1)

    Y_train = (
        y_train[:, None] > thresholds[None, :]
    ).astype(np.float64)

    # --------------------------------------------------
    # 3. Center targets
    # --------------------------------------------------
    y_mean = Y_train.mean(axis=0)

    Y_centered = Y_train - y_mean

    # --------------------------------------------------
    # 4. Ridge regression using SVD
    # --------------------------------------------------
    U, s, Vt = np.linalg.svd(
        X_train,
        full_matrices=False,
    )

    if ridge == 0:
        cutoff = (
            np.finfo(float).eps
            * max(X_train.shape)
            * s[0]
        )

        factors = np.divide(
            1.0,
            s,
            out=np.zeros_like(s),
            where=s > cutoff,
        )

    else:
        factors = s / (s**2 + ridge)

    W = (Vt.T * factors) @ (U.T @ Y_centered)

    # --------------------------------------------------
    # 5. Ordinal predictions
    # --------------------------------------------------
    train_scores = X_train @ W + y_mean
    test_scores = X_test @ W + y_mean

    train_pred = np.sum(
        train_scores > 0.5,
        axis=1,
    ).astype(np.int64)

    test_pred = np.sum(
        test_scores > 0.5,
        axis=1,
    ).astype(np.int64)

    # --------------------------------------------------
    # 6. Metrics
    # --------------------------------------------------
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)

    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "train_ser": float(1.0 - train_acc),
        "test_ser": float(1.0 - test_acc),
        "weight_norm": float(np.linalg.norm(W)),
        "weight_max": float(np.max(np.abs(W))),
        "feature_dim": int(X.shape[1]),
        "W": W,
        "train_pred": train_pred,
        "test_pred": test_pred,
    }



def build_parity_aware_features(X, symbol_indices=None):
    """
    Construct parity-aware reservoir features.

    X_parity[k] = [
        X[k],
        (-1)^k * X[k]
    ]

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Reservoir features, optionally after temporal windowing.

    symbol_indices : ndarray, optional
        Original symbol indices corresponding to each row of X.
        If None, assumes row k corresponds to symbol k.

    Returns
    -------
    X_parity : ndarray, shape (N, 2*D)
    """

    X = np.asarray(X, dtype=float)

    N, D = X.shape

    if symbol_indices is None:
        symbol_indices = np.arange(N)

    symbol_indices = np.asarray(symbol_indices)

    if symbol_indices.shape != (N,):
        raise ValueError(
            "symbol_indices must have shape (N,)"
        )

    parity = (
        1.0 - 2.0 * (symbol_indices % 2)
    ).reshape(-1, 1)

    X_parity = np.concatenate(
        [
            X,
            parity * X,
        ],
        axis=1,
    )

    print("\nParity-aware features")
    print("Original dimension:", D)
    print("New dimension:", X_parity.shape[1])

    return X_parity


def build_parity_only_features(X, symbol_indices=None):

    X = np.asarray(X, dtype=float)

    N = X.shape[0]

    if symbol_indices is None:
        symbol_indices = np.arange(N)

    parity = (
        1.0 - 2.0 * (
            np.asarray(symbol_indices) % 2
        )
    )[:, None]

    return parity * X
# ============================================================
# 4. Common Train/Test Indices
# ============================================================

def common_indices(
    n_symbols,
    max_L,
):
    """
    Build common train/test indices.

    Future windows must not cross the train/test
    boundary or the final valid channel edge.
    """

    valid_start = max(
        WASHOUT,
        CHANNEL_RADIUS,
    )

    valid_stop = (
        n_symbols - CHANNEL_RADIUS
    )

    valid_indices = np.arange(
        valid_start,
        valid_stop,
    )

    split_pos = int(
        len(valid_indices) * TRAIN_RATIO
    )

    train_boundary = int(
        valid_indices[split_pos]
    )

    train_idx = valid_indices[:split_pos]

    test_idx = valid_indices[split_pos:]

    # Prevent future-window leakage
    train_idx = train_idx[
        train_idx + max_L < train_boundary
    ]

    test_idx = test_idx[
        test_idx + max_L < valid_stop
    ]

    if (
        len(train_idx) == 0
        or len(test_idx) == 0
    ):
        raise ValueError(
            "Empty train/test split."
        )

    return train_idx, test_idx


# ============================================================
# 5. Channel Diagnostics
# ============================================================

'''def channel_diagnostics(
    task,
    strength,
):
    """
    Check nonlinear channel monotonicity.

    Channel:
        z = q + s*(a*q^2 + b*q^3)

    Derivative:
        dz/dq = 1 + s*(2*a*q + 3*b*q^2)

    The check uses observed q samples only.
    """

    q = np.asarray(
        task.metadata["linear_channel_output"],
        dtype=float,
    )

    q = q[
        CHANNEL_RADIUS:
        len(q)-CHANNEL_RADIUS
    ]

    derivative = (
        1
        + strength * (
            2 * BASE_QUAD * q
            + 3 * BASE_CUBIC * q**2
        )
    )

    derivative_min = float(
        derivative.min()
    )

    derivative_max = float(
        derivative.max()
    )

    nonmonotone_fraction = float(
        np.mean(derivative <= 0)
    )

    return (
        derivative_min,
        derivative_max,
        nonmonotone_fraction,
    )'''
def channel_diagnostics(task, strength):
    """
    Analyze nonlinear channel monotonicity.

    Channel:
        z = q + s*(a*q^2 + b*q^3)

    Derivative:
        dz/dq = 1 + s*(2*a*q + 3*b*q^2)

    Checks:
        - Minimum derivative
        - Maximum derivative
        - Fraction of observed samples with derivative <= 0
        - Whether the continuous mapping has a turning point
          within the observed q range
    """

    q = np.asarray(
        task.metadata["linear_channel_output"],
        dtype=float,
    )

    q = q[
        CHANNEL_RADIUS:
        len(q) - CHANNEL_RADIUS
    ]

    q_min = float(np.min(q))
    q_max = float(np.max(q))

    a = strength * BASE_QUAD
    b = strength * BASE_CUBIC

    def derivative(x):
        return (
            1.0
            + 2.0 * a * x
            + 3.0 * b * x**2
        )

    # --------------------------------------------------------
    # Observed derivative statistics
    # --------------------------------------------------------

    observed_derivative = derivative(q)

    fraction_nonpositive = float(
        np.mean(observed_derivative <= 0)
    )

    # --------------------------------------------------------
    # Continuous derivative range
    #
    # dz/dq is quadratic in q.
    # Its extrema occur at:
    #   - interval endpoints
    #   - the quadratic vertex, if inside the interval
    # --------------------------------------------------------

    candidate_q = [q_min, q_max]

    if abs(b) > 1e-15:

        vertex = -a / (3.0 * b)

        if q_min <= vertex <= q_max:
            candidate_q.append(vertex)

    derivative_values = np.array(
        [derivative(x) for x in candidate_q],
        dtype=float,
    )

    derivative_min = float(
        np.min(derivative_values)
    )

    derivative_max = float(
        np.max(derivative_values)
    )

    # Strict sign change implies a turning point
    # within the observed amplitude interval.
    has_turning_point = bool(
        derivative_min < 0.0
        and derivative_max > 0.0
    )

    return {
        "q_min": q_min,
        "q_max": q_max,
        "derivative_min": derivative_min,
        "derivative_max": derivative_max,
        "fraction_nonpositive": fraction_nonpositive,
        "has_turning_point": has_turning_point,
    }


# ============================================================
# 6. Main Experiment
# ============================================================

def main():

    print("=" * 70)
    print("Nonlinear PAM4 Distortion Strength Sweep")
    print("=" * 70)

    max_L = max(PT_WINDOWS)

    train_idx, test_idx = common_indices(
        N_SYMBOLS,
        max_L,
    )

    print(
        f"T_SYMBOL = {T_SYMBOL*1e9:.4f} ns"
    )

    print(
        f"dt = "
        f"{T_SYMBOL/POINTS_PER_SYMBOL*1e12:.4f} ps"
    )

    print(f"K = {K}")

    print(
        f"Train samples = {len(train_idx)}"
    )

    print(
        f"Test samples = {len(test_idx)}"
    )

    print(
        f"RUN_RESERVOIR = {RUN_RESERVOIR}"
    )

    print(
        f"AWGN = {ADD_AWGN}"
    )

    print("=" * 70)

    rows = []

    # --------------------------------------------------------
    # Sweep distortion strength
    # --------------------------------------------------------

    for strength in DISTORTION_STRENGTHS:

        quadratic = (
            strength * BASE_QUAD
        )

        cubic = (
            strength * BASE_CUBIC
        )

        print()
        print("=" * 70)

        print(
            f"Distortion strength = {strength}"
        )

        print(
            f"Channel: "
            f"z = q + ({quadratic:.6f})q^2 "
            f"+ ({cubic:.6f})q^3"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Generate nonlinear PAM4 task
        # ----------------------------------------------------

        task = generate_pam4_nonlinear_task(
            n_symbols=N_SYMBOLS,
            seed=SEED,
            washout=WASHOUT,
            train_ratio=TRAIN_RATIO,

            quadratic=quadratic,
            cubic=cubic,

            add_awgn=ADD_AWGN,
            snr_db=SNR_DB,
            noise_seed=NOISE_SEED,

            channel_radius=CHANNEL_RADIUS,
        )

        u = np.asarray(
            task.u,
            dtype=float,
        )

        y = np.asarray(
            task.y,
            dtype=int,
        )

        # ----------------------------------------------------
        # Channel diagnostics
        # ----------------------------------------------------

        '''(
            derivative_min,
            derivative_max,
            nonmonotone_fraction,
        ) = channel_diagnostics(
            task,
            strength,
        )'''
        diagnostics = channel_diagnostics(
            task,
            strength,
        )

        derivative_min = diagnostics["derivative_min"]

        derivative_max = diagnostics["derivative_max"]

        nonmonotone_fraction = diagnostics[
            "fraction_nonpositive"
        ]

        has_turning_point = diagnostics[
            "has_turning_point"
        ]

        print(
            f"Derivative min = "
            f"{derivative_min:.6f}"
        )

        print(
            f"Derivative max = "
            f"{derivative_max:.6f}"
        )

        print(
            f"Fraction derivative <= 0 = "
            f"{nonmonotone_fraction:.4%}"
        )

        # ----------------------------------------------------
        # Input-only features
        # ----------------------------------------------------

        X_input = u.reshape(-1, 1)

        # ----------------------------------------------------
        # 5-tap linear equalizer
        # ----------------------------------------------------

        X_5tap = np.full(
            (len(y), 5),
            np.nan,
            dtype=float,
        )

        valid_n = np.arange(
            CHANNEL_RADIUS,
            len(y)-CHANNEL_RADIUS,
        )

        for j, offset in enumerate(
            range(-2, 3)
        ):

            X_5tap[
                valid_n,
                j,
            ] = u[
                valid_n + offset
            ]

        # ----------------------------------------------------
        # Baseline experiments
        # ----------------------------------------------------

        '''experiments = [
            (
                "input_only",
                None,
                X_input,
            ),
            (
                "linear_5tap",
                None,
                X_5tap,
            ),
        ]'''
        # ----------------------------------------------------
        # Polynomial equalizer features
        # ----------------------------------------------------

        X_poly2 = polynomial_features(
            X_5tap,
            degree=2,
        )

        X_poly3 = polynomial_features(
            X_5tap,
            degree=3,
        )

        # ----------------------------------------------------
        # Baseline experiments
        # ----------------------------------------------------

        experiments = [
            (
                "input_only",
                None,
                X_input,
            ),
            (
                "linear_5tap",
                None,
                X_5tap,
            ),
            (
                "polynomial_5tap_deg2",
                None,
                X_poly2,
            ),
            (
                "polynomial_5tap_deg3",
                None,
                X_poly3,
            ),
        ]

        # ----------------------------------------------------
        # Reservoir simulation
        # ----------------------------------------------------

        if RUN_RESERVOIR:

            reservoir = PTSingleReservoir(
                eta=1.0,
                alpha=ALPHA,
            )

            sim = simulate_continuous_task(
                reservoir=reservoir,
                task_data=task,

                symbol_period=T_SYMBOL,
                points_per_symbol=POINTS_PER_SYMBOL,
                k=K,

                amplitude_scale=I_SCALE,
                #encoding="zoh",
                solver = 'ivp',
                encoding = 'carrier',
                carrier_frequency=48e6,
                reset_phase=False,
                #pulse_duty=PULSE_DUTY_VALUE,
            )

            X = np.asarray(
                sim.X,
                dtype=float,
            )

            expected_shape = (
                len(y),
                2 * K,
            )

            if X.shape != expected_shape:

                raise ValueError(
                    f"Unexpected reservoir shape: "
                    f"{X.shape}, "
                    f"expected {expected_shape}"
                )
            print("\nReservoir feature statistics")
            print("X shape:", X.shape)
            print("X min:", np.min(X))
            print("X max:", np.max(X))
            print("X mean:", np.mean(X))
            print("X std:", np.std(X))
            print("NaN:", np.isnan(X).sum())
            print("Inf:", np.isinf(X).sum())

            # ------------------------------------------------
            # Full temporal windows
            # ------------------------------------------------

            '''for L in PT_WINDOWS:

                X_window = build_future_window(
                    X,
                    L,
                )

                experiments.append(
                    (
                        "pt_full_window",
                        L,
                        X_window,
                    )
                )'''

            for L in PT_WINDOWS:

                X_window = build_future_window(
                    X,
                    L,
                )

                # Original reservoir features
                experiments.append(
                    (
                        "pt_full_window",
                        L,
                        X_window,
                    )
                )

                # Parity-aware reservoir features
                X_parity = build_parity_aware_features(
                    X_window,
                    symbol_indices=np.arange(len(y)),
                )

                experiments.append(
                    (
                        "pt_parity_aware",
                        L,
                        X_parity,
                    )
                )

                X_parity_only = build_parity_only_features(
                    X_window,
                    symbol_indices=np.arange(len(y)),
                )

                experiments.append(
                    (
                        "pt_parity_only",
                        L,
                        X_parity_only,
                    )
                )

        # ----------------------------------------------------
        # Evaluate all methods
        # ----------------------------------------------------

        for method, L, features in experiments:

            metrics = evaluate(
                features,
                y,
                train_idx,
                test_idx,
                ridge=RIDGE,
            )

            multiclass_result = evaluate_multiclass(
                features,
                y,
                train_idx,
                test_idx,
                ridge=RIDGE,
            )

            ordinal_result = evaluate_ordinal(
                features,
                y,
                train_idx,
                test_idx,
                ridge=RIDGE,
            )

            print("Classes:", np.unique(y))

            print("-" * 60)

            print(
                f"Scalar Ridge Accuracy:     "
                f"{metrics['test_accuracy']:.4f}"
            )

            print(
                f"Multiclass Ridge Accuracy: "
                f"{multiclass_result['test_accuracy']:.4f}"
            )

            print(
                f"Ordinal Ridge Accuracy:    "
                f"{ordinal_result['test_accuracy']:.4f}"
            )

            print(
                f"Ordinal W norm:            "
                f"{ordinal_result['weight_norm']:.4e}"
            )

            print(
                f"Ordinal W max:             "
                f"{ordinal_result['weight_max']:.4e}"
            )
            
            

            print("\nOrdinal Ridge Confusion Matrix:")

            cm = confusion_matrix(
                y[test_idx],
                ordinal_result["test_pred"],
                labels=np.arange(4),
            )

            print(cm)

            print("\nPer-class Recall:")

            recall = np.diag(cm) / np.maximum(
                cm.sum(axis=1),
                1,
            )

            for c, r in enumerate(recall):
                print(f"Class {c}: {r:.4%}")




            row = {
                "strength": strength,

                "quadratic": quadratic,
                "cubic": cubic,

                "awgn": ADD_AWGN,

                "snr_db": (
                    SNR_DB if ADD_AWGN else ""
                ),

                "derivative_min": derivative_min,

                "derivative_max": derivative_max,

                "nonmonotone_fraction":
                    nonmonotone_fraction,

                "has_turning_point": has_turning_point,

                "method": method,

                "L": (
                    "" if L is None else L
                ),

                "latency_ns": (
                    2 * T_SYMBOL * 1e9
                    if method in (
                        "linear_5tap",
                        "polynomial_5tap_deg2",
                        "polynomial_5tap_deg3",
                    )
                    else (
                        0.0
                        if L is None
                        else L * T_SYMBOL * 1e9
                    )
                ),
                "latency_ns": (
                    0.0
                    if L is None
                    else L * T_SYMBOL * 1e9
                ),

                **metrics,
            }

            rows.append(row)

            print(
                f"{method:<18} "
                f"L={str(L):>3} "
                f"D={metrics['feature_dim']:>4} "
                f"Test acc="
                f"{metrics['test_accuracy']:.2%} "
                f"Test NRMSE="
                f"{metrics['test_nrmse']:.4f}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if rows:

        with OUTPUT_CSV.open(
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys()),
            )

            writer.writeheader()

            writer.writerows(rows)

        print()
        print("=" * 70)

        print(
            f"Saved: {OUTPUT_CSV.resolve()}"
        )

        print("=" * 70)


if __name__ == "__main__":
    main()