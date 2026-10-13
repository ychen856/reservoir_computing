import numpy as np
from src.tasks.base import TaskData


PAM4_LEVELS = np.array(
    [-3.0, -1.0, 1.0, 3.0]
)


def generate_pam4(
    n_symbols=3000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    labels = rng.integers(
        0,
        4,
        size=n_symbols
    )

    symbols = PAM4_LEVELS[labels]

    return symbols, labels




def generate_pam4_task(
    n_symbols=3000,
    seed=42,
    washout=200,
    train_ratio=0.8,
):
    """
    PAM4 classification through a linear ISI channel.

    Input:
        Normalized channel output q[n].

    Target:
        PAM4 class index c[n] in {0, 1, 2, 3}.

    Readout protocol:
        Scalar ridge regression followed by quantization.
    """

    if n_symbols <= washout + 2:
        raise ValueError(
            "n_symbols must be greater than washout + 2"
        )

    symbols, labels = generate_pam4(
        n_symbols=n_symbols,
        seed=seed,
    )

    channel_output = apply_linear_channel(symbols)

    input_scale = np.max(np.abs(channel_output))

    if input_scale == 0:
        raise ValueError("PAM4 channel output has zero amplitude")

    u = channel_output / input_scale

    return TaskData(
        name="pam4_linear",
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
            "channel_output": channel_output,
            "input_scale": input_scale,
            "readout_type": "scalar_ridge_quantization",
            "simulation_mode": "continuous_sequence",
            'channel_radius': 2,
        },
    )

def apply_linear_channel(symbols):
    """
    Linear ISI channel.

    q[n] =
        0.08 * a[n-2]
      - 0.12 * a[n-1]
      + 1.00 * a[n]
      + 0.18 * a[n+1]
      - 0.10 * a[n+2]
    """

    a = np.asarray(symbols, dtype=float)

    q = np.zeros_like(a)

    for n in range(2, len(a) - 2):
        q[n] = (
              0.08 * a[n - 2]
            - 0.12 * a[n - 1]
            + 1.00 * a[n]
            + 0.18 * a[n + 1]
            - 0.10 * a[n + 2]
        )

    return q


def quantize_pam4_class(y_pred):
    return np.clip(
        np.rint(y_pred),
        0,
        3
    ).astype(int)

def build_channel_window(u, radius=2):
    X = np.zeros(
        (len(u), 2 * radius + 1)
    )

    for n in range(
        radius,
        len(u) - radius
    ):
        X[n] = u[
            n-radius:n+radius+1
        ]

    return X