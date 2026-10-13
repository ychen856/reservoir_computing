import numpy as np


def generate_narma10(
    n_samples=3000,
    seed=42
):
    """
    Generate the standard NARMA-10 benchmark.

    Input:
        u[n] ~ Uniform(0, 0.5)

    Target:
        y[n+1] =
            0.3*y[n]
            + 0.05*y[n]*sum(y[n-i], i=0..9)
            + 1.5*u[n-9]*u[n]
            + 0.1

    Returns
    -------
    u : ndarray, shape (n_samples,)
    y : ndarray, shape (n_samples,)
    """

    rng = np.random.default_rng(seed)

    u = rng.uniform(
        0.0,
        0.5,
        size=n_samples
    )

    y = np.zeros(n_samples)

    for n in range(9, n_samples - 1):

        y_history = np.sum(
            y[n-9:n+1]
        )

        y[n+1] = (
            0.3 * y[n]
            + 0.05 * y[n] * y_history
            + 1.5 * u[n-9] * u[n]
            + 0.1
        )

    return u, y





