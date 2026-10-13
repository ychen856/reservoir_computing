import numpy as np

def nrmse(y_true, y_pred):

    mse = np.mean(
        (y_true - y_pred)**2
    )

    return np.sqrt(
        mse / np.var(y_true)
    )


def nmse(y_true, y_pred):
    return nrmse(y_true, y_pred) ** 2


def pearson_r(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def accuracy(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


def symbol_error_rate(y_true, y_pred):
    return 1.0 - accuracy(y_true, y_pred)