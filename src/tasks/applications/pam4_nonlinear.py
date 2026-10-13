import numpy as np

from src.tasks.base import TaskData
from src.tasks.applications.pam4 import (
    generate_pam4,
    apply_linear_channel,
    PAM4_LEVELS,
)


def apply_nonlinear_distortion(
    q,
    quadratic=0.036,
    cubic=-0.011,
):
    """
    Nonlinear channel distortion:

        d[n] = q[n] + quadratic*q[n]^2 + cubic*q[n]^3
    """
    q = np.asarray(q, dtype=float)

    return (
        q
        + quadratic * q**2
        + cubic * q**3
    )


def generate_pam4_nonlinear_task(
    n_symbols=3000,
    seed=42,
    washout=200,
    train_ratio=0.8,
    quadratic=0.036,
    cubic=-0.011,
    add_awgn=False,
    snr_db=40.0,
    noise_seed=123,
    channel_radius=2,
):
    """
    PAM4 classification through linear ISI + nonlinear distortion.

    Input:
        Normalized distorted received waveform.

    Target:
        PAM4 class index in {0, 1, 2, 3}.

    Readout protocol:
        Scalar ridge regression followed by quantization.
    """

    if n_symbols <= washout + 2 * channel_radius:
        raise ValueError(
            "Not enough symbols after washout and channel boundaries"
        )

    # --------------------------------------------------------
    # PAM4 symbols
    # --------------------------------------------------------

    symbols, labels = generate_pam4(
        n_symbols=n_symbols,
        seed=seed,
    )

    # --------------------------------------------------------
    # Linear ISI channel
    # --------------------------------------------------------

    q_raw = apply_linear_channel(symbols)

    # --------------------------------------------------------
    # Nonlinear distortion
    # --------------------------------------------------------

    d_raw = apply_nonlinear_distortion(
        q_raw,
        quadratic=quadratic,
        cubic=cubic,
    )

    # --------------------------------------------------------
    # Optional AWGN
    # --------------------------------------------------------

    u_raw = d_raw.copy()

    measured_snr_db = None
    noise_power = 0.0

    if add_awgn:
        rng = np.random.default_rng(noise_seed)

        valid_signal = d_raw[
            channel_radius:-channel_radius
        ]

        signal_power = np.mean(valid_signal**2)

        noise_power = signal_power / (
            10.0 ** (snr_db / 10.0)
        )

        noise = rng.normal(
            loc=0.0,
            scale=np.sqrt(noise_power),
            size=len(d_raw),
        )

        u_raw = d_raw + noise

        measured_noise_power = np.mean(
            noise[channel_radius:-channel_radius] ** 2
        )

        measured_snr_db = 10.0 * np.log10(
            signal_power / measured_noise_power
        )

    # --------------------------------------------------------
    # Normalize received waveform
    # --------------------------------------------------------

    input_scale = np.max(np.abs(u_raw))

    if input_scale == 0:
        raise ValueError(
            "PAM4 received waveform has zero amplitude"
        )

    u = u_raw / input_scale

    # --------------------------------------------------------
    # TaskData
    # --------------------------------------------------------

    return TaskData(
        name="pam4_nonlinear",
        u=u,
        y=labels,
        task_type="classification",
        washout=washout,
        train_ratio=train_ratio,
        target_offset=0,
        metadata={
            "seed": seed,
            "levels": PAM4_LEVELS.copy(),
            "symbols": symbols,
            "linear_channel_output": q_raw,
            "distorted_channel_output": d_raw,
            "received_waveform": u_raw,
            "input_scale": input_scale,
            "quadratic": quadratic,
            "cubic": cubic,
            "add_awgn": add_awgn,
            "snr_db": snr_db,
            "noise_seed": noise_seed,
            "noise_power": noise_power,
            "measured_snr_db": measured_snr_db,
            "channel_radius": channel_radius,
            "readout_type": "scalar_ridge_quantization",
            "simulation_mode": "continuous_sequence",
        },
    )