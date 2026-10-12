import numpy as np


def generate_sine_cube(
    n_samples=2000,
    n_periods=20,
    amplitude=1.0,
):
    """
    Generate the sine-to-cube benchmark.

    Input:
        u(t) = A sin(2*pi*f*t)

    Target:
        y(t) = u(t)^3

    Returns
    -------
    t : ndarray, shape (n_samples,)
        Normalized task time.

    u : ndarray, shape (n_samples,)
        Input sequence.

    y : ndarray, shape (n_samples,)
        Target sequence.
    """

    t = np.linspace(
        0.0,
        n_periods,
        n_samples,
        endpoint=False
    )

    u = amplitude * np.sin(2.0 * np.pi * t)

    y = u**3

    return t, u, y