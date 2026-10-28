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

from src.readout.metrics import nrmse, pearson_r
from src.tasks.applications.pam4 import quantize_pam4_class



# ============================================================
# Evaluation result
# ============================================================

@dataclass
class RegressionResult:

    train_nrmse: float
    test_nrmse: float

    train_nrmse_per_output: np.ndarray
    test_nrmse_per_output: np.ndarray
    
    train_r_per_output: np.ndarray
    test_r_per_output: np.ndarray

    y_train: np.ndarray
    y_test: np.ndarray

    y_train_pred: np.ndarray
    y_test_pred: np.ndarray

    X_train: np.ndarray
    X_test: np.ndarray

    W: np.ndarray

    mu: np.ndarray
    std: np.ndarray

    n_train: int
    n_test: int

    feature_dim: int
    ridge: float


# ============================================================
# Continuous regression evaluation
# ============================================================

def evaluate_continuous_regression(
    X,
    y,
    washout=0,
    train_ratio=0.8,
    ridge=1e-6,
    normalize=True,
):
    """
    Evaluate a continuous-sequence regression task.

    Pipeline
    --------
    1. Washout
    2. Chronological train/test split
    3. Train-only standardization
    4. Ridge regression
    5. NRMSE evaluation

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Reservoir features.

    y : ndarray, shape (N,) or (N, O)
        Regression targets.

    washout : int
        Number of initial samples discarded.

    train_ratio : float
        Fraction of remaining samples used for training.

    ridge : float
        Ridge regularization coefficient.

    normalize : bool
        Whether to standardize features using training data.

    Returns
    -------
    RegressionResult
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if X.ndim != 2:
        raise ValueError(
            "X must have shape (N, D)"
        )

    if y.ndim not in (1, 2):
        raise ValueError(
            "y must have shape (N,) or (N, O)"
        )

    if len(X) != len(y):
        raise ValueError(
            "Feature/target length mismatch"
        )

    if not 0 <= washout < len(X):
        raise ValueError(
            "Invalid washout"
        )

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1"
        )

    if ridge < 0:
        raise ValueError(
            "ridge must be non-negative"
        )

    # --------------------------------------------------------
    # Washout
    # --------------------------------------------------------

    X_eval = X[washout:]
    y_eval = y[washout:]

    # --------------------------------------------------------
    # Chronological train/test split
    # --------------------------------------------------------

    n_train = int(train_ratio * len(X_eval))

    if n_train == 0 or n_train == len(X_eval):
        raise ValueError(
            "Train/test split produced an empty partition"
        )

    X_train = X_eval[:n_train]
    y_train = y_eval[:n_train]

    X_test = X_eval[n_train:]
    y_test = y_eval[n_train:]

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    if normalize:

        mu, std = fit_standardizer(X_train)

        X_train_n = transform_standardizer(
            X_train,
            mu,
            std,
        )

        X_test_n = transform_standardizer(
            X_test,
            mu,
            std,
        )

    else:

        mu = np.zeros(X.shape[1])
        std = np.ones(X.shape[1])

        X_train_n = X_train.copy()
        X_test_n = X_test.copy()

    # --------------------------------------------------------
    # Ridge regression
    # --------------------------------------------------------

    W = train_ridge_regression(
        X_train_n,
        y_train,
        ridge=ridge,
    )

    y_train_pred = predict_ridge_regression(
        X_train_n,
        W,
    )

    y_test_pred = predict_ridge_regression(
        X_test_n,
        W,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    train_nrmse = nrmse(
        y_train,
        y_train_pred,
    )

    test_nrmse = nrmse(
        y_test,
        y_test_pred,
    )

    # ============================================================
    # Per-output metrics
    # ============================================================

    Y_train_2d = np.asarray(y_train).reshape(len(y_train), -1)
    Y_test_2d = np.asarray(y_test).reshape(len(y_test), -1)

    P_train_2d = np.asarray(y_train_pred).reshape(len(y_train), -1)
    P_test_2d = np.asarray(y_test_pred).reshape(len(y_test), -1)

    n_outputs = Y_train_2d.shape[1]

    train_nrmse_per_output = np.array([
        nrmse(Y_train_2d[:, j], P_train_2d[:, j])
        for j in range(n_outputs)
    ])

    test_nrmse_per_output = np.array([
        nrmse(Y_test_2d[:, j], P_test_2d[:, j])
        for j in range(n_outputs)
    ])

    train_r_per_output = np.array([
        pearson_r(Y_train_2d[:, j], P_train_2d[:, j])
        for j in range(n_outputs)
    ])

    test_r_per_output = np.array([
        pearson_r(Y_test_2d[:, j], P_test_2d[:, j])
        for j in range(n_outputs)
    ])
    

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return RegressionResult(
        train_nrmse=float(train_nrmse),
        test_nrmse=float(test_nrmse),

        train_nrmse_per_output=train_nrmse_per_output,
        test_nrmse_per_output=test_nrmse_per_output,

        train_r_per_output=train_r_per_output,
        test_r_per_output=test_r_per_output,

        y_train=y_train,
        y_test=y_test,

        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,

        X_train=X_train_n,
        X_test=X_test_n,

        W=W,

        mu=mu,
        std=std,

        n_train=len(y_train),
        n_test=len(y_test),

        feature_dim=X.shape[1],
        ridge=ridge,
    )





@dataclass
class ClassificationResult:

    train_accuracy: float
    test_accuracy: float

    train_ser: float
    test_ser: float

    confusion_matrix: np.ndarray

    y_train: np.ndarray
    y_test: np.ndarray

    y_train_pred: np.ndarray
    y_test_pred: np.ndarray

    X_train: np.ndarray
    X_test: np.ndarray

    W: np.ndarray

    mu: np.ndarray
    std: np.ndarray

    n_train: int
    n_test: int

    feature_dim: int
    n_classes: int
    ridge: float



def evaluate_independent_classification(
    X_train,
    y_train,
    X_test,
    y_test,
    ridge=1e-6,
    normalize=True,
    n_classes=None,
):
    """
    Evaluate independent-trial multiclass classification.

    Pipeline
    --------
    1. Validate train/test datasets
    2. Train-only feature standardization
    3. One-hot ridge regression
    4. Argmax class prediction
    5. Accuracy / SER / confusion matrix

    Parameters
    ----------
    X_train : ndarray, shape (N_train, D)
    y_train : ndarray, shape (N_train,)
    X_test : ndarray, shape (N_test, D)
    y_test : ndarray, shape (N_test,)
    ridge : float
        Ridge regularization coefficient.
    normalize : bool
        Whether to standardize using training statistics.
    n_classes : int or None
        Number of classes. Labels must be 0, ..., n_classes-1.

    Returns
    -------
    ClassificationResult
    """

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            "X_train and X_test must have shape (N, D)"
        )

    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError(
            "y_train and y_test must be 1D arrays"
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training feature/label length mismatch"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Testing feature/label length mismatch"
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Train/test feature dimensions do not match"
        )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Training and testing datasets must be nonempty"
        )

    if ridge < 0:
        raise ValueError(
            "ridge must be non-negative"
        )

    if n_classes is None:
        n_classes = int(max(
            np.max(y_train),
            np.max(y_test),
        )) + 1

    if n_classes < 2:
        raise ValueError(
            "At least two classes are required"
        )

    if (
        np.any(y_train < 0)
        or np.any(y_train >= n_classes)
        or np.any(y_test < 0)
        or np.any(y_test >= n_classes)
    ):
        raise ValueError(
            "Labels must be in [0, n_classes-1]"
        )

    # --------------------------------------------------------
    # Train-only standardization
    # --------------------------------------------------------

    if normalize:

        mu, std = fit_standardizer(X_train)

        X_train_n = transform_standardizer(
            X_train,
            mu,
            std,
        )

        X_test_n = transform_standardizer(
            X_test,
            mu,
            std,
        )

    else:

        mu = np.zeros(X_train.shape[1])
        std = np.ones(X_train.shape[1])

        X_train_n = X_train.copy()
        X_test_n = X_test.copy()

    # --------------------------------------------------------
    # One-hot targets
    # --------------------------------------------------------

    Y_train = np.eye(n_classes)[y_train]

    # --------------------------------------------------------
    # Ridge classifier
    # --------------------------------------------------------

    W = train_ridge_regression(
        X_train_n,
        Y_train,
        ridge=ridge,
    )

    train_scores = predict_ridge_regression(
        X_train_n,
        W,
    )

    test_scores = predict_ridge_regression(
        X_test_n,
        W,
    )

    y_train_pred = np.argmax(
        train_scores,
        axis=1,
    )

    y_test_pred = np.argmax(
        test_scores,
        axis=1,
    )

    # --------------------------------------------------------
    # Accuracy / SER
    # --------------------------------------------------------

    train_accuracy = np.mean(
        y_train_pred == y_train
    )

    test_accuracy = np.mean(
        y_test_pred == y_test
    )

    train_ser = 1.0 - train_accuracy
    test_ser = 1.0 - test_accuracy

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = np.zeros(
        (n_classes, n_classes),
        dtype=int,
    )

    for actual, predicted in zip(
        y_test,
        y_test_pred,
    ):
        cm[actual, predicted] += 1

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return ClassificationResult(
        train_accuracy=float(train_accuracy),
        test_accuracy=float(test_accuracy),

        train_ser=float(train_ser),
        test_ser=float(test_ser),

        confusion_matrix=cm,

        y_train=y_train,
        y_test=y_test,

        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,

        X_train=X_train_n,
        X_test=X_test_n,

        W=W,

        mu=mu,
        std=std,

        n_train=len(y_train),
        n_test=len(y_test),

        feature_dim=X_train.shape[1],
        n_classes=n_classes,
        ridge=ridge,
    )


@dataclass
class ScalarClassificationResult:

    train_nrmse: float
    test_nrmse: float

    train_accuracy: float
    test_accuracy: float

    train_ser: float
    test_ser: float

    confusion_matrix: np.ndarray

    y_train: np.ndarray
    y_test: np.ndarray

    y_train_pred: np.ndarray
    y_test_pred: np.ndarray

    y_train_class: np.ndarray
    y_test_class: np.ndarray

    X_train: np.ndarray
    X_test: np.ndarray

    W: np.ndarray

    mu: np.ndarray
    std: np.ndarray

    n_train: int
    n_test: int

    feature_dim: int
    ridge: float


def evaluate_scalar_classification(
    X,
    y,
    washout=0,
    train_ratio=0.8,
    ridge=1e-6,
    normalize=True,
    valid_start=None,
    valid_stop=None,
    quantize_fn=quantize_pam4_class,
    n_classes=4,
):
    """
    Continuous scalar regression followed by class quantization.

    Suitable for PAM4 linear/nonlinear channel benchmarks.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Input or reservoir features.

    y : ndarray, shape (N,)
        Integer class labels, e.g. 0, 1, 2, 3.

    valid_start, valid_stop : int or None
        Valid evaluation interval [start, stop).
        Used to exclude channel boundary samples.

    Returns
    -------
    ScalarClassificationResult
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if X.ndim != 2:
        raise ValueError("X must have shape (N, D)")

    if y.ndim != 1:
        raise ValueError("y must be 1D")

    if len(X) != len(y):
        raise ValueError("Feature/target length mismatch")

    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    if not 0 < train_ratio < 1:
        raise ValueError("Invalid train_ratio")

    if not 0 <= washout < len(X):
        raise ValueError("Invalid washout")

    if valid_start is None:
        start = washout
    else:
        start = max(washout, valid_start)

    if valid_stop is None:
        stop = len(X)
    else:
        stop = min(valid_stop, len(X))

    if stop <= start:
        raise ValueError("No valid evaluation samples")

    X_eval = X[start:stop]
    y_eval = y[start:stop]

    n_train = int(train_ratio * len(X_eval))

    if n_train == 0 or n_train == len(X_eval):
        raise ValueError("Empty train/test partition")

    X_train = X_eval[:n_train]
    X_test = X_eval[n_train:]

    y_train = y_eval[:n_train]
    y_test = y_eval[n_train:]

    # ========================================================
    # Train-only standardization
    # ========================================================

    if normalize:

        mu, std = fit_standardizer(X_train)

        X_train_n = transform_standardizer(
            X_train, mu, std
        )

        X_test_n = transform_standardizer(
            X_test, mu, std
        )

    else:

        mu = np.zeros(X.shape[1])
        std = np.ones(X.shape[1])

        X_train_n = X_train.copy()
        X_test_n = X_test.copy()

    # ========================================================
    # Scalar ridge regression
    # ========================================================

    W = train_ridge_regression(
        X_train_n,
        y_train,
        ridge=ridge,
    )

    y_train_pred = np.asarray(
        predict_ridge_regression(X_train_n, W)
    ).reshape(-1)

    y_test_pred = np.asarray(
        predict_ridge_regression(X_test_n, W)
    ).reshape(-1)

    # ========================================================
    # PAM4 symbol decision
    # ========================================================

    y_train_class = quantize_fn(y_train_pred)
    y_test_class = quantize_fn(y_test_pred)

    # ========================================================
    # Metrics
    # ========================================================

    train_nrmse = nrmse(y_train, y_train_pred)
    test_nrmse = nrmse(y_test, y_test_pred)

    train_accuracy = np.mean(
        y_train_class == y_train
    )

    test_accuracy = np.mean(
        y_test_class == y_test
    )

    train_ser = 1.0 - train_accuracy
    test_ser = 1.0 - test_accuracy

    cm = np.zeros(
        (n_classes, n_classes),
        dtype=int,
    )

    for actual, predicted in zip(
        y_test,
        y_test_class,
    ):
        cm[actual, predicted] += 1

    return ScalarClassificationResult(
        train_nrmse=float(train_nrmse),
        test_nrmse=float(test_nrmse),

        train_accuracy=float(train_accuracy),
        test_accuracy=float(test_accuracy),

        train_ser=float(train_ser),
        test_ser=float(test_ser),

        confusion_matrix=cm,

        y_train=y_train,
        y_test=y_test,

        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,

        y_train_class=y_train_class,
        y_test_class=y_test_class,

        X_train=X_train_n,
        X_test=X_test_n,

        W=W,

        mu=mu,
        std=std,

        n_train=len(y_train),
        n_test=len(y_test),

        feature_dim=X.shape[1],
        ridge=ridge,
    )