import numpy as np
from src.utils.sampling import sample_reservoir_states

def sample_reservoir_voltages(sol, sample_times):
    """
    Compatibility wrapper for V1/V2 sampling.
    """
    return sample_reservoir_states(
        sol,
        sample_times,
        state_indices=(0, 1),
    )

def extract_temporal_features(
    sol,
    n_symbols,
    symbol_period,
    k=8,
    delay_symbols=0,
    state_indices=(0, 1),
):
    """
    Extract K temporal nodes per symbol from selected reservoir states.

    Returns
    -------
    X : ndarray, shape (n_valid, k * len(state_indices))
    """

    if k < 1:
        raise ValueError("k must be >= 1")

    if delay_symbols < 0:
        raise ValueError("delay_symbols must be >= 0")

    if len(state_indices) == 0:
        raise ValueError("state_indices cannot be empty")

    n_valid = n_symbols - delay_symbols

    if n_valid <= 0:
        raise ValueError("No valid symbols after delay")

    X = np.zeros((n_valid, k * len(state_indices)))

    for n in range(n_valid):

        t_start = (n + delay_symbols) * symbol_period
        t_stop = (n + delay_symbols + 1) * symbol_period

        sample_times = np.linspace(
            t_start,
            t_stop,
            k + 2,
        )[1:-1]

        X[n] = sample_reservoir_states(
            sol,
            sample_times,
            state_indices=state_indices,
        )

    return X

def extract_temporal_voltage_feature(
    sol,
    K,
    t_start,
    t_end,
):

    sample_times = np.linspace(
        t_start,
        t_end,
        K,
    )
    feature = sample_reservoir_voltages(
        sol,
        sample_times,
    )

    return sample_times, feature



'''def build_mpsk_feature_matrix(
    sols,
    K,
    t_start,
    t_end
):

    features = []

    sample_times = None

    for sol in sols:

        sample_times, feature = (
            extract_temporal_voltage_feature(
                sol=sol,
                K=K,
                t_start=t_start,
                t_end=t_end
            )
        )

        features.append(feature)

    X = np.asarray(features)

    return sample_times, X'''





def build_repeated_noise_dataset(
    sols_by_class,
    K,
    t_start,
    t_end
):

    X = []
    y = []

    sample_times = None

    for m, class_sols in enumerate(sols_by_class):

        for sol in class_sols:
            print(sol)
            sample_times, feature = (
                extract_temporal_voltage_feature(
                    sol=sol,
                    K=K,
                    t_start=t_start,
                    t_end=t_end
                )
            )

            X.append(feature)
            y.append(m)

    X = np.asarray(X)
    y = np.asarray(y)

    return sample_times, X, y


def build_mpsk_feature_matrix(
    sols,
    K,
    t_start,
    t_end,
):
    """
    Extract temporal voltage features from independent
    MPSK reservoir simulations.

    Parameters
    ----------
    sols : list[OdeResult]
        One reservoir solution per trial.

    Returns
    -------
    sample_times : ndarray, shape (K,)
    X : ndarray, shape (N_trials, 2*K)
    """

    features = []

    sample_times = np.linspace(
        t_start,
        t_end,
        K,
    )

    for sol in sols:
        feature = sample_reservoir_voltages(
            sol,
            sample_times,
        )

        features.append(feature.reshape(-1))

    X = np.asarray(features)

    return sample_times, X


def reorder_features_by_age(
    X,
    n_pairs,
    k,
    symbol_indices=None,
):
    """
    Reorder time-major parallel features by injection age.

    Assumptions
    -----------
    - Interleaved routing: symbol n drives pair n % n_pairs.
    - Original columns:
      [time0: pair0 V1,V2, pair1 V1,V2, ...,
       time1: pair0 V1,V2, pair1 V1,V2, ...]

    Output
    ------
    Same shape and temporal ordering, with pair blocks ordered as:
    [currently driven pair, previous pair, ...].
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if (
        not isinstance(n_pairs, (int, np.integer))
        or n_pairs < 1
        or not isinstance(k, (int, np.integer))
        or k < 1
    ):
        raise ValueError("n_pairs and k must be positive integers")

    expected_dim = 2 * n_pairs * k

    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} features, got {X.shape[1]}"
        )

    n_samples = len(X)

    if symbol_indices is None:
        symbol_indices = np.arange(n_samples)
    else:
        symbol_indices = np.asarray(symbol_indices)

        if symbol_indices.shape != (n_samples,):
            raise ValueError(
                "symbol_indices must contain one index per row"
            )

        if (
            not np.issubdtype(symbol_indices.dtype, np.integer)
            or np.any(symbol_indices < 0)
        ):
            raise ValueError(
                "symbol_indices must be nonnegative integers"
            )

    # Shape: (samples, temporal nodes, physical pairs, voltages)
    blocks = X.reshape(n_samples, k, n_pairs, 2)

    # Age 0 = currently driven pair.
    # Age 1 = pair driven in the previous symbol.
    physical_pairs = (
        symbol_indices[:, None] - np.arange(n_pairs)[None, :]
    ) % n_pairs

    reordered = np.take_along_axis(
        blocks,
        physical_pairs[:, None, :, None],
        axis=2,
    )

    return reordered.reshape(X.shape)