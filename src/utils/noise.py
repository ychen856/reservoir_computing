import numpy as np

def add_awgn(
    signal,
    snr_db,
    rng,
    power_mask=None
):
    """
    Add AWGN to a sampled clean waveform.

    Parameters
    ----------
    signal : ndarray
        Clean waveform samples.

    snr_db : float
        Desired SNR in dB.

    rng : np.random.Generator
        Random-number generator.

    power_mask : ndarray of bool, optional
        Region used to estimate signal power.
        Noise is still added to the whole waveform.

    Returns
    -------
    noisy_signal : ndarray
    noise : ndarray
    signal_power : float
    noise_power_target : float
    """

    signal = np.asarray(signal)

    if power_mask is None:
        signal_for_power = signal
    else:
        signal_for_power = signal[power_mask]

    signal_power = np.mean(signal_for_power**2)

    noise_power_target = (
        signal_power / (10.0**(snr_db / 10.0))
    )

    noise_std = np.sqrt(noise_power_target)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=signal.shape
    )

    noisy_signal = signal + noise

    return (
        noisy_signal,
        noise,
        signal_power,
        noise_power_target
    )
