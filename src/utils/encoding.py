import numpy as np


def sample_hold_input(
    sequence,
    symbol_period,
    amplitude_scale=1.0
):
    """
    Convert a discrete task sequence into a continuous-time
    zero-order-hold input function.
    """
    if symbol_period <= 0:
        raise ValueError("symbol_period must be positive")

    sequence = np.asarray(sequence, dtype=float)

    def input_func(t):
        idx = int(np.floor(t / symbol_period))

        if idx < 0:
            return 0.0

        if idx >= len(sequence):
            return 0.0

        return amplitude_scale * sequence[idx]

    return input_func


def carrier_encoded_input(
    sequence,
    symbol_period,
    carrier_frequency,
    amplitude_scale=1.0
):
    if symbol_period <= 0:
        raise ValueError("symbol_period must be positive")
    
    sequence = np.asarray(sequence, dtype=float)

    def input_func(t):

        idx = int(np.floor(t / symbol_period))

        if idx < 0 or idx >= len(sequence):
            return 0.0

        u = sequence[idx]

        return (
            amplitude_scale
            * u
            * np.cos(
                2.0 * np.pi
                * carrier_frequency
                * t
            )
        )

    return input_func