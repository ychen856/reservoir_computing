import numpy as np


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