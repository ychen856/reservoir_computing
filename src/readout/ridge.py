import numpy as np

def train_linear_readout(X, y, n_classes, ridge=1e-6):
    """
    Ridge-regression linear readout.

    X: (N, D)
    y: (N,)
    """

    N = X.shape[0]

    # Add bias
    Xb = np.column_stack([
        X,
        np.ones(N)
    ])

    # One-hot target
    Y = np.eye(n_classes)[y]

    # Ridge solution
    I = np.eye(Xb.shape[1])

    # Usually don't regularize bias
    I[-1, -1] = 0.0

    Wout = np.linalg.solve(
        Xb.T @ Xb + ridge * I,
        Xb.T @ Y
    )

    return Wout


def predict_linear_readout(X, Wout):

    Xb = np.column_stack([
        X,
        np.ones(X.shape[0])
    ])

    scores = Xb @ Wout

    y_pred = np.argmax(scores, axis=1)

    return y_pred, scores



def train_ridge_regression(X, y, ridge=1e-6):
    X_aug = np.column_stack([
        np.ones(X.shape[0]),
        X
    ])

    reg = np.eye(X_aug.shape[1])
    reg[0, 0] = 0.0

    W = np.linalg.solve(
        X_aug.T @ X_aug + ridge * reg,
        X_aug.T @ y
    )

    return W


def predict_ridge_regression(X, W):
    X_aug = np.column_stack([
        np.ones(X.shape[0]),
        X
    ])

    return X_aug @ W




def nrmse(y_true, y_pred):

    mse = np.mean(
        (y_true - y_pred)**2
    )

    return np.sqrt(
        mse / np.var(y_true)
    )

