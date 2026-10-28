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


def bipolar_input(
    sequence,
    symbol_period,
    amplitude_scale=1.0,
    mode="peak_matched",
):
    """
    Convert a discrete sequence into a bipolar ZOH input.

    Assumes sequence values are in [0, 0.5].

    Modes:
        centered:
            u - 0.25
        peak_matched:
            2*u - 0.5
        full_scale:
            4*u - 1
    """
    if symbol_period <= 0:
        raise ValueError("symbol_period must be positive")

    sequence = np.asarray(sequence, dtype=float)

    if mode == "centered":
        encoded = sequence - 0.25
    elif mode == "peak_matched":
        encoded = 2.0 * sequence - 0.5
    elif mode == "full_scale":
        encoded = 4.0 * sequence - 1.0
    else:
        raise ValueError(f"Unknown bipolar mode: {mode}")

    def input_func(t):
        idx = int(np.floor(t / symbol_period))

        if idx < 0 or idx >= len(encoded):
            return 0.0

        return amplitude_scale * encoded[idx]

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


def masked_zoh_input(
    sequence,
    symbol_period,
    amplitude_scale=1.0,
    mask=(1.0, -1.0),
):
    """
    Piecewise-constant masked input.

    Each original symbol is divided into M segments.
    The mask is repeated for every symbol.
    """
    if symbol_period <= 0:
        raise ValueError("symbol_period must be positive")

    sequence = np.asarray(sequence, dtype=float)
    mask = np.asarray(mask, dtype=float)

    if mask.ndim != 1 or len(mask) == 0:
        raise ValueError("mask must be a nonempty 1D array")

    M = len(mask)
    segment_period = symbol_period / M

    def input_func(t):
        segment_idx = int(np.floor(t / segment_period))

        if segment_idx < 0:
            return 0.0

        symbol_idx = segment_idx // M

        if symbol_idx >= len(sequence):
            return 0.0

        mask_idx = segment_idx % M

        return (
            amplitude_scale
            * sequence[symbol_idx]
            * mask[mask_idx]
        )

    return input_func
