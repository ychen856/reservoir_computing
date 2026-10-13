import numpy as np
from src.tasks.base import TaskData


def generate_henon(
    n_samples=3000,
    a=1.4,
    b=0.3,
    x0=0.1,
    y0=0.0,
    transient=1000,
    washout=200,
    train_ratio=0.8,
):
    """
    Generate Hénon observer benchmark.

    Dynamics:
        x[n+1] = 1 - a*x[n]^2 + y[n]
        y[n+1] = b*x[n]

    Task:
        input  = x[n]
        target = y[n]

    The physical input is normalized to [-1, 1]
    using the maximum absolute value of x.
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

    x = x[start:stop]
    y = y[start:stop]

    # Normalize physical reservoir input only.
    # Keep the target in its original scale.
    u_scale = np.max(np.abs(x))

    if u_scale == 0:
        raise ValueError("Hénon input has zero amplitude")

    u = x / u_scale

    return TaskData(
        name="henon",
        u=u,
        y=y,
        task_type="regression",
        washout=washout,
        train_ratio=train_ratio,
        target_offset=0,
        metadata={
            "a": a,
            "b": b,
            "transient": transient,
            "input_scale": u_scale,
            "raw_x": x,
            "simulation_mode": "continuous_sequence",
        },
    )
