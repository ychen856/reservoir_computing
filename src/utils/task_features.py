import numpy as np


def extract_temporal_features_temp(
    sol,
    n_symbols,
    symbol_period,
    k=8,
):
    """
    Extract K temporal samples of V1 and V2
    from each task symbol.
    """

    t = sol.t
    V1 = sol.y[0]
    V2 = sol.y[1]

    X = []

    for n in range(n_symbols):

        t_start = n * symbol_period
        t_stop = (n + 1) * symbol_period

        sample_times = np.linspace(
            t_start,
            t_stop,
            k + 2
        )[1:-1]

        features = []

        for ts in sample_times:
            idx = np.argmin(np.abs(t - ts))

            features.extend([
                V1[idx],
                V2[idx]
            ])

        X.append(features)

    return np.asarray(X)


def extract_temporal_features(
    sol,
    n_symbols,
    symbol_period,
    k=8,
    delay_symbols=0,
):
    """
    Extract K temporal nodes from V1 and V2 for each symbol.

    delay_symbols = 0:
        sample during symbol n

    delay_symbols = 1:
        sample during symbol n+1

    delay_symbols = 2:
        sample during symbol n+2
    """

    n_valid = n_symbols - delay_symbols

    X = np.zeros(
        (n_valid, 2 * k)
    )

    for n in range(n_valid):

        t_start = (
            n + delay_symbols
        ) * symbol_period

        t_stop = (
            n + delay_symbols + 1
        ) * symbol_period

        sample_times = np.linspace(
            t_start,
            t_stop,
            k + 2
        )[1:-1]

        V1_samples = np.interp(
            sample_times,
            sol.t,
            sol.y[0]
        )

        V2_samples = np.interp(
            sample_times,
            sol.t,
            sol.y[1]
        )

        X[n, 0::2] = V1_samples
        X[n, 1::2] = V2_samples

    return X


def normalize_train_test(X_train, X_test):

    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std < 1e-12] = 1.0

    X_train_n = (X_train - mu) / std
    X_test_n = (X_test - mu) / std

    return X_train_n, X_test_n
