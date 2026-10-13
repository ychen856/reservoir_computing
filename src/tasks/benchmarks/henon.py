import numpy as np


def generate_henon(n_samples=3000, a=1.4, b=0.3, x0=0.1, y0=0.0,
                   transient=1000):
    """Generate a Hénon-map trajectory after discarding a transient.

    x[n+1] = 1 - a*x[n]^2 + y[n]
    y[n+1] = b*x[n]
    """
    total = n_samples + transient
    x = np.empty(total + 1, dtype=float)
    y = np.empty(total + 1, dtype=float)
    x[0], y[0] = x0, y0

    for n in range(total):
        x[n + 1] = 1.0 - a * x[n] ** 2 + y[n]
        y[n + 1] = b * x[n]

    start = transient + 1
    stop = start + n_samples
    return x[start:stop], y[start:stop]
